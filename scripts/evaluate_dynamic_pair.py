"""Evaluate existing policies only; validate moving RTX geometry and replay first."""
import argparse,hashlib,json,subprocess,sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root);protocol=json.loads((root/'protocol.json').read_text())
    if (root/'status.json').exists():raise ValueError('Use a fresh output directory; existing runs are preserved')
    source=Path(__file__)
    (root/'runner_source.py').write_bytes(source.read_bytes())
    hashes={name:sha(path) for name,path in protocol['policies'].items()}
    commands=[]
    def status(state,stage,**extra):
        data={'state':state,'stage':stage,'updated_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'policy_sha256':hashes,'commands':commands,**extra}
        (root/'status.json').write_text(json.dumps(data,indent=2)+'\n')
        print(state,stage,flush=True)
    def run(stage,command):
        occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if occupied:raise RuntimeError('GPU has other compute processes: '+occupied)
        commands.append({'stage':stage,'argv':command})
        status('running',stage)
        with (root/(stage+'.log')).open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        run('render_probe',[sys.executable,'scripts/check_dynamic_render.py','--out',str(root/'render_probe')])
        replay={}
        for name,path in protocol['policies'].items():
            for suffix in ['smoke','smoke_replay']:
                stage=name+'_'+suffix
                cmd=[sys.executable,'scripts/run.py','evaluate','--config',protocol['config'],'--policy',path,
                    '--out',str(root/stage),'--episodes','3','--num-envs','16','--seed',str(protocol['evaluation_seed'])]
                if name=='current_only':cmd.append('--current-only')
                run(stage,cmd)
            replay[name]={filename:sha(root/(name+'_smoke')/filename)==sha(root/(name+'_smoke_replay')/filename)
                for filename in ['episodes.json','trajectories.json','scenes.json']}
            (root/'smoke_replay_checks.json').write_text(json.dumps(replay,indent=2))
            if not all(replay[name].values()):raise RuntimeError('Dynamic smoke replay mismatch: '+name)
        for name,path in protocol['policies'].items():
            stage=name+'_full'
            cmd=[sys.executable,'scripts/run.py','evaluate','--config',protocol['config'],'--policy',path,
                '--out',str(root/stage),'--episodes',str(protocol['episodes_per_density']),
                '--num-envs',str(protocol['num_envs']),'--seed',str(protocol['evaluation_seed'])]
            if name=='current_only':cmd.append('--current-only')
            run(stage,cmd)
        scenes=[json.loads((root/(name+'_full')/'scenes.json').read_text()) for name in protocol['policies']]
        if scenes[0]!=scenes[1]:raise RuntimeError('Full evaluation scene/track manifests differ')
        if hashes!={name:sha(path) for name,path in protocol['policies'].items()}:raise RuntimeError('Policy weights changed')
        status('completed','evaluations_finished',smoke_replay_checks=replay,full_scenes_equal=True)
    except Exception as error:
        status('failed','see_logs',error=repr(error));raise


if __name__=='__main__':main()
