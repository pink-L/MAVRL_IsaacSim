"""Paired 30% / 50% evaluation of existing policies, with no training stages."""
import argparse,hashlib,json,subprocess,sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root);protocol=json.loads((root/'protocol.json').read_text())
    if (root/'status.json').exists():raise ValueError('Existing run must be preserved; use a new root')
    if not json.loads((root/'geometry_audit.json').read_text())['all_passed']:raise ValueError('Geometry audit has not passed')
    (root/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    hashes={name:sha(path) for name,path in protocol['policies'].items()};commands=[];replay={}
    def status(state,stage,**extra):
        (root/'status.json').write_text(json.dumps({'state':state,'stage':stage,
            'updated_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'policy_sha256':hashes,
            'commands':commands,**extra},indent=2)+'\n');print(state,stage,flush=True)
    def run(percent,name,suffix,episodes):
        occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if occupied:raise RuntimeError('GPU already occupied: '+occupied)
        stage=f'pct{percent}_{name}_{suffix}'
        command=[sys.executable,'scripts/run.py','evaluate','--config',protocol['configs'][str(percent)],
            '--policy',protocol['policies'][name],'--out',str(root/stage),'--episodes',str(episodes),
            '--num-envs',str(protocol['num_envs']),'--seed',str(protocol['evaluation_seed'])]
        if name=='current_only':command.append('--current-only')
        commands.append({'stage':stage,'argv':command});status('running',stage)
        with (root/(stage+'.log')).open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        # The 50% condition exercises the maximum moving-geometry load.
        for name in protocol['policies']:
            run(50,name,'smoke',3);run(50,name,'smoke_replay',3)
            replay[name]={f:sha(root/f'pct50_{name}_smoke'/f)==sha(root/f'pct50_{name}_smoke_replay'/f)
                for f in ['episodes.json','trajectories.json','scenes.json']}
            (root/'smoke_replay_checks.json').write_text(json.dumps(replay,indent=2)+'\n')
            if not all(replay[name].values()):raise RuntimeError('High-fraction replay mismatch: '+name)
        for percent,name in protocol['run_order']:
            run(percent,name,'full',protocol['episodes_per_density'])
        manifests={}
        for percent in [30,50]:
            values=[json.loads((root/f'pct{percent}_{name}_full/scenes.json').read_text()) for name in protocol['policies']]
            if values[0]!=values[1]:raise ValueError('Models did not see identical map/motion tasks')
            manifests[percent]=values[0]
        for key,a in manifests[30].items():
            b=manifests[50][key];k=len(a['moving']['indices'])
            if a['original_cylinders']!=b['original_cylinders']:raise ValueError('Ratio conditions changed the original map')
            if any(value!=b['moving'][field][:k] for field,value in a['moving'].items()):raise ValueError('30% is not an exact motion subset of 50%')
        if hashes!={name:sha(path) for name,path in protocol['policies'].items()}:raise ValueError('Policy weights changed')
        status('completed','evaluations_finished',smoke_replay_checks=replay,nested_motion_verified=True)
    except Exception as error:
        status('failed','see_logs',error=repr(error));raise


if __name__=='__main__':main()
