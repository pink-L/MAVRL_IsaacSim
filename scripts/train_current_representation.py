"""Run the offline current-only LSTM while simulator repeatability is investigated.

Uses the baseline's existing data, VAE and initial policy. Does not launch PPO,
change the baseline failure status, or declare the complete ablation finished.
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
    p.add_argument('--baseline', required=True)
    p.add_argument('--root', required=True)
    a = p.parse_args()
    baseline, root = Path(a.baseline).resolve(), Path(a.root)
    checks = json.loads((baseline/'validation.json').read_text())
    required = ['finite_losses', 'representation_exactly_frozen', 'separate_critic_present',
                'checkpoint_load_action_identity', 'train_validation_scene_disjoint']
    if not all(checks.get(k) for k in required):
        raise ValueError('Baseline artifacts did not pass the required offline checks')
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                       '--format=csv,noheader,nounits'], text=True).strip()
    if occupied:
        raise RuntimeError('GPU occupied: '+occupied)
    cfg = json.loads((baseline/'lstm/config.json').read_text())
    cfg.pop('_run', None)
    files = [baseline/'initial/latest.pt', baseline/'dataset/manifest.json', baseline/'vae/best.pt']
    hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    root.mkdir(parents=True, exist_ok=False)
    folder = root/'long_run'
    folder.mkdir()
    for name in ['initial', 'dataset', 'vae']:
        (root/name).symlink_to(baseline/name, target_is_directory=True)
    (root/'REUSED_INPUTS.json').write_text(json.dumps({
        'baseline': str(baseline), 'sha256': hashes,
        'baseline_status_at_start': json.loads((baseline/'long_run/status.json').read_text()),
        'scope': 'Paired offline representation ablation with shared data/VAE/initial policy. '
                 'Baseline capture discrepancy remains unresolved; training data retain their original provenance.'}, indent=2)+'\n')
    (root/'representation_input_config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    command = ['scripts/python.sh', 'scripts/run.py', 'lstm', '--current-only',
               '--config', str(root/'representation_input_config.json'),
               '--dataset', str(root/'dataset'), '--vae', str(root/'vae/best.pt'),
               '--policy', str(root/'initial/latest.pt'), '--out', str(root/'lstm')]
    state = {'pid': os.getpid(), 'state': 'running', 'stage': 'lstm', 'current_only': True,
             'command': command, 'scope': 'Offline LSTM only; PPO and evaluation remain outstanding.'}

    def update(**values):
        state.update(values, updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
        (folder/'status.json').write_text(json.dumps(state, indent=2)+'\n')

    try:
        with (folder/'lstm.log').open('xb') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            update(child_pid=child.pid)
            code = child.wait()
        if code:
            raise RuntimeError(f'LSTM exited with {code}')
        if not (root/'lstm/memory_diagnostic.json').exists():
            raise RuntimeError('Missing post-training diagnostic')
        update(state='representation_ready', child_pid=None, last_completed_stage='lstm')
    except BaseException as exc:
        update(state='failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
