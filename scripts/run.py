"""Unified reproducible entry point. Every simulator experiment uses Isaac RTX."""
import argparse,json,copy,shutil,subprocess,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

def resolved_config(args):
    checkpoint=None
    if args.resume:
        import torch
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False)
        if checkpoint.get('phase') not in ('initial','final'):
            raise ValueError('Resume requires an initial/final PPO training checkpoint')
        if args.phase is not None and args.phase!=checkpoint['phase']:
            raise ValueError('Resume phase conflicts with checkpoint; use --representation for a new final run')
        args.phase=checkpoint['phase']
    args.phase=args.phase or 'initial'
    if args.config:
        cfg=json.loads(Path(args.config).read_text())
    elif checkpoint:
        cfg=copy.deepcopy(checkpoint['config'])
    else:
        cfg=json.loads(Path('configs/isaac_mavrl.json').read_text())
    # --steps is an early stopping budget. Keep the official LR schedule unless
    # explicitly overridden; the original suggested 200-iteration bootstrap is
    # an early stop of the 8e6-step schedule, not a decay-to-zero in 200 iterations.
    cfg['ppo'].setdefault('schedule_timesteps',cfg['ppo']['total_timesteps'])
    for key in ['seed','num_envs']:
        if getattr(args,key) is not None:cfg[key]=getattr(args,key)
    for arg,key in [('steps','total_timesteps'),('n_steps','n_steps'),('batch_size','batch_size'),('warmup_steps','warmup_timesteps'),('schedule_steps','schedule_timesteps')]:
        if getattr(args,arg) is not None:cfg['ppo'][key]=getattr(args,arg)
    if args.epochs is not None:cfg['vae']['epochs']=cfg['lstm']['epochs']=args.epochs
    if getattr(args,'sequence_length',None) is not None:cfg['lstm']['sequence_length']=args.sequence_length
    if args.current_only:cfg['lstm']['reconstruction_members']=[0,1,0]
    return cfg

def record_run(out,cfg,args):
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    folder=out/'source'/stamp
    files=sorted(Path('mavrl_isaac').glob('*.py'))+sorted(Path('scripts').glob('*.py'))
    hashes={}
    for file in files:
        target=folder/file;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
        hashes[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    cfg['_run']={'source_snapshot':str(folder),'source_sha256':hashes,'git_commit':commit,'git_dirty':dirty}
    metadata={'command':vars(args),'config':cfg}
    (out/f'run_{stamp}.json').write_text(json.dumps(metadata,indent=2))
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    (out/'command.json').write_text(json.dumps(vars(args),indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['train','collect','vae','lstm','evaluate'])
    p.add_argument('--config');p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int);p.add_argument('--num-envs',type=int)
    p.add_argument('--phase',choices=['initial','final'])
    p.add_argument('--steps',type=int);p.add_argument('--n-steps',type=int);p.add_argument('--batch-size',type=int)
    p.add_argument('--warmup-steps',type=int);p.add_argument('--epochs',type=int)
    p.add_argument('--schedule-steps',type=int)
    p.add_argument('--policy');p.add_argument('--representation');p.add_argument('--resume')
    p.add_argument('--dataset');p.add_argument('--vae');p.add_argument('--frames',type=int,default=4096)
    p.add_argument('--sequence-length',type=int)
    p.add_argument('--episodes',type=int,default=20);p.add_argument('--densities',nargs='+',default=['sparse','medium','dense'])
    p.add_argument('--reset-memory',action='store_true');p.add_argument('--current-only',action='store_true')
    a=p.parse_args();cfg=resolved_config(a)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    record_run(out,cfg,a)
    from mavrl_isaac.training import seed_all
    seed_all(cfg['seed'])
    if a.command in ('vae','lstm'):
        from mavrl_isaac.representation_training import train_vae,train_lstm
        if a.command=='vae':train_vae(cfg,a.dataset,out)
        else:train_lstm(cfg,a.dataset,a.vae,a.policy,out)
        return
    from mavrl_isaac.renderer import launch,IsaacDepthRenderer
    app=launch(runtime_log=out/'isaac_runtime.log')
    try:
        from mavrl_isaac.environment import NavigationEnv
        renderer=IsaacDepthRenderer(cfg);env=NavigationEnv(cfg,renderer)
        if a.command=='train':
            from mavrl_isaac.training import train_ppo
            train_ppo(env,cfg,out,a.phase,a.representation,a.resume)
        elif a.command=='collect':
            from mavrl_isaac.dataset import collect
            collect(env,a.policy,cfg,out,a.frames)
        else:
            from mavrl_isaac.evaluation import evaluate
            evaluate(env,cfg,a.policy,out,a.episodes,a.densities,a.reset_memory)
    finally:app.close()

if __name__=='__main__':main()
