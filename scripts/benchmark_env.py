import argparse,json,time,subprocess,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--num-envs',type=int,default=4);p.add_argument('--steps',type=int,default=100);p.add_argument('--tiled',action='store_true');p.add_argument('--out',default='artifacts');p.add_argument('--policy');p.add_argument('--warmup',type=int,default=0);args=p.parse_args()
out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
from mavrl_isaac.renderer import launch,IsaacDepthRenderer
app=launch()
try:
 import numpy as np
 from PIL import Image
 from mavrl_isaac.environment import NavigationEnv
 cfg=json.load(open('configs/isaac_mavrl.json'));cfg['num_envs']=args.num_envs;cfg['camera']['tiled']=args.tiled
 renderer=IsaacDepthRenderer(cfg);env=NavigationEnv(cfg,renderer)
 obs=env.reset()
 if args.policy:
  import torch
  from mavrl_isaac.training import load_checkpoint,seed_all
  seed_all(cfg['seed']);rep,actor,_=load_checkpoint(args.policy,cfg['device']);actor.eval()
  h=rep.zero_state(env.n);hv=rep.zero_state(env.n);starts=np.ones(env.n,bool)
 def advance():
  global obs,h,hv,starts
  if args.policy:
   with torch.no_grad():
    z,zv,h,hv=rep.step_both(obs['image'],h,hv,starts)
    state=torch.as_tensor(obs['state'],device=cfg['device'])
    action,_,_=actor.act(torch.cat([z,state],-1),deterministic=True,critic_x=torch.cat([zv,state],-1))
   obs,rew,starts,info=env.step(action.cpu().numpy())
  else:obs,rew,done,info=env.step(np.zeros((args.num_envs,4)))
  return info
 for i in range(args.num_envs):Image.fromarray(obs['image'][i,0]).save(out/f'depth_env_{args.num_envs}_{i}.png')
 for _ in range(args.warmup):advance()
 episodes=0
 start=time.perf_counter()
 for _ in range(args.steps):
  info=advance();episodes+=sum('episode' in item for item in info)
 seconds=time.perf_counter()-start
 result={'config':cfg,'tiled':args.tiled,'num_envs':args.num_envs,'steps':args.steps,'seconds':seconds,'transitions_per_second':args.steps*args.num_envs/seconds,
   'policy':args.policy,'warmup':args.warmup,'completed_episodes':episodes,'scope':'Rollout with source actor/critic recurrent features and actual scene resets; excludes PPO gradient updates' if args.policy else 'Zero-action static camera/environment stepping',
   'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
   'process_pid':os.getpid(),'gpu_processes':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True).strip(),
   'image_min':int(obs['image'].min()),'image_max':int(obs['image'].max()),'image_std':float(obs['image'].std()),
   'scenes':[{'seed':s.seed,'density':s.density,'obstacles':len(s.cylinders),'traversability':s.traversability} for s in env.scenes]}
 (out/f"benchmark_{args.num_envs}{'_tiled' if args.tiled else str()}.json").write_text(json.dumps(result,indent=2))
 print('BENCHMARK',json.dumps(result),flush=True)
finally:app.close()
