"""Qualify a capture setting, then run paired PPOs with that identical setting.

All inputs remain provenance-labelled. A failed capture check stops dependent
training and preserves diagnostics. No old run is relabelled as completed.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--config', default='configs/isaac_capture2.json')
    p.add_argument('--resume', action='store_true', help='Reuse qualified captures and fully completed branches; never overwrite partial training')
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=a.resume)
    if a.resume:
        previous=json.loads((out/'status.json').read_text())
        if previous.get('pid') and Path(f"/proc/{previous['pid']}").exists():raise RuntimeError('Recorded supervisor PID is still present')
        stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
        (out/f'status_before_resume_{stamp}.json').write_text(json.dumps(previous,indent=2)+'\n')
        qualification=json.loads((out/'capture_qualification.json').read_text())
        if qualification['config_sha256']!=hashlib.sha256(Path(a.config).read_bytes()).hexdigest():raise ValueError('Capture configuration changed; use a new qualification run')
    baseline = Path('artifacts/formal_seed0')
    current = Path('artifacts/ablation_current_seed0')
    probe = baseline/'policy_capture_probe_subframes2'
    state = {'pid': os.getpid(), 'state': 'starting', 'config': a.config}

    def update(**values):
        state.update(values, updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        (out/'status.json').write_text(json.dumps(state, indent=2)+'\n')

    def run(stage, command, gpu=False):
        if gpu:
            occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                               '--format=csv,noheader,nounits'], text=True).strip()
            if occupied: raise RuntimeError('GPU occupied before '+stage+': '+occupied)
        update(state='running', stage=stage, command=command)
        with (out/f'{stage}.log').open('xb') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            update(child_pid=child.pid)
            code = child.wait()
        if code: raise RuntimeError(f'{stage} exited with {code}')
        update(child_pid=None, last_completed_stage=stage)

    try:
        check = json.loads((probe/'status.json').read_text())
        if check['state'] != 'completed' or check['differing_capture_pairs'] != 0 or check['captures'] < 3000:
            raise ValueError('Capture settling probe did not pass; dependent training remains stopped')
        reports = []
        for index in [1, 2]:
            target = out/f'evaluation_{index}'
            if a.resume:
                saved=json.loads((target/'summary.json').read_text())
                if saved['sha256']!=hashlib.sha256((baseline/'final/latest.pt').read_bytes()).hexdigest():raise ValueError('Qualification policy changed')
            else:
                run(f'evaluation_{index}', ['scripts/python.sh', 'scripts/run.py', 'evaluate',
                    '--config', a.config, '--policy', str(baseline/'final/latest.pt'),
                    '--out', str(target), '--episodes', '100', '--num-envs', '16', '--seed', '200'], True)
            reports.append(target)
        left, right = reports
        comparisons = {name: (left/name).read_bytes() == (right/name).read_bytes()
                       for name in ['episodes.json', 'trajectories.json']}
        comparisons['summary_results_equal'] = (json.loads((left/'summary.json').read_text())['results']
                                                == json.loads((right/'summary.json').read_text())['results'])
        qualification = {'probe': check, 'repeat_comparisons': comparisons,
                         'config_sha256': hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
                         'scope': 'Observed same-state and repeated-policy checks; not a proof of determinism on every future frame or device.'}
        (out/'capture_qualification.json').write_text(json.dumps(qualification, indent=2)+'\n')
        if not all(comparisons.values()):
            raise ValueError('Candidate capture setting failed independent policy replay')
        update(capture_qualified=True)
        # Matching capture configuration is part of a controlled reconstruction
        # ablation: retrain both PPO heads, retaining the old baseline as history.
        roots = []
        for variant, inputs, members in [('baseline', baseline, [1, 1, 0]),
                                         ('current', current, [0, 1, 0]),
                                         ('fixed_speed', baseline, [1, 1, 0])]:
            target = Path(f'artifacts/capture2_{variant}_seed0')
            if a.resume and target.exists():
                completed=json.loads((target/'long_run/status.json').read_text())
                if completed['state']!='completed':raise ValueError('Existing branch is incomplete; recover it before resuming: '+str(target))
                checks=json.loads((target/'validation.json').read_text())
                if not all(checks.get(k) for k in ['finite_losses','representation_exactly_frozen','checkpoint_load_action_identity','separate_critic_present']):raise ValueError('Completed branch lacks validation')
                roots.append(target);update(last_reused_branch=variant);continue
            target.mkdir(exist_ok=False)
            files = [inputs/'initial/latest.pt', inputs/'dataset/manifest.json',
                     inputs/'vae/best.pt', inputs/'lstm/best.pt']
            for name in ['initial', 'dataset', 'vae', 'lstm']:
                (target/name).symlink_to((inputs/name).resolve(), target_is_directory=True)
            (target/'REUSED_INPUTS.json').write_text(json.dumps({
                'source': str(inputs), 'reconstruction_members': members,
                'sha256': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                'scope': 'Fresh seed=0 PPO using the same qualified capture setting in both branches. '
                         'Data/VAE/LSTM are shared or reused historical artifacts; their source capture configuration is preserved.'}, indent=2)+'\n')
            roots.append(target)
            config = 'configs/isaac_capture2_fixed_speed.json' if variant == 'fixed_speed' else a.config
            command = ['scripts/python.sh', 'scripts/run_long_pipeline.py', '--formal', '--prepared-root',
                       '--initial-policy', str(target/'initial/latest.pt'), '--root', str(target),
                       '--config', config, '--steps', '8000000', '--episodes', '100', '--num-envs', '64']
            if variant == 'current': command.append('--current-only')
            run(f'{variant}_pipeline', command, True)
        run('paired_comparison', ['scripts/python.sh', 'scripts/compare_memory_ablation.py',
                                  str(roots[0]), str(roots[1]), '--out', str(out/'memory_comparison')])
        run('speed_comparison', ['scripts/python.sh', 'scripts/compare_speed_ablation.py',
                                str(roots[0]), str(roots[2]), '--out', str(out/'speed_comparison')])
        update(state='experiments_complete', stage='finished',
               remaining='Review evidence and update the final report; capture/data provenance limitations remain explicit.')
    except BaseException as exc:
        update(state='failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
