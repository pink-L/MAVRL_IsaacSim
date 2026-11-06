"""Isolate Warp/Torch copy races from renderer settling on real changing scenes."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=160);p.add_argument('--subframes',type=int);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
cfg=json.loads(Path('configs/isaac_mavrl.json').read_text());cfg['num_envs']=16
from mavrl_isaac.renderer import launch,IsaacDepthRenderer
app=launch()
try:
 import warp as wp
 import carb
 from mavrl_isaac.environment import NavigationEnv
 renderer=IsaacDepthRenderer(cfg);env=NavigationEnv(cfg,renderer);original=renderer.view.get_depth;copy_races=[]
 if a.subframes is not None:
  original_step=renderer.rep.orchestrator.step
  def fixed_step(**kwargs):kwargs['rt_subframes']=a.subframes;return original_step(**kwargs)
  renderer.rep.orchestrator.step=fixed_step
 def checked_depth(*args,**kwargs):
  depth=original(*args,**kwargs);before=depth.cpu().numpy().copy();wp.synchronize_device('cuda:0');after=depth.cpu().numpy().copy()
  different=~np.isclose(before,after,rtol=0,atol=0,equal_nan=True)
  if different.any():copy_races.append({'call':len(calls),'pixels':int(different.sum())})
  calls.append(1);return depth
 calls=[];renderer.view.get_depth=checked_depth
 obs=env.reset();settling=[];rng=np.random.default_rng(71)
 for i in range(a.steps):
  if i%4==0:
   ids=rng.choice(env.n,size=4,replace=False)
   for j in ids:env.reset_one(j,int(rng.integers(100000,200000)))
  obs,_,_,_=env.step(rng.uniform(-1,1,(env.n,4)))
  # No state/geometry changes: repeated capture is a render-settling probe.
  again=env.observe();delta=abs(obs['image'].astype(float)-again['image'].astype(float))
  if delta.max()>2:
   settling.append({'step':i,'mae_gray':float(delta.mean()),'pixels_gt_2':int((delta>2).sum()),'max_abs_gray':float(delta.max())})
 result={'steps':a.steps,'forced_subframes':a.subframes,'copy_races':copy_races,'static_repeated_capture_differences':settling,'warp_version':wp.__version__,
  'warp_source':wp.__file__,'async_rendering':carb.settings.get_settings().get('/app/asyncRendering'),
  'renderer_sha256':hashlib.sha256(Path('mavrl_isaac/renderer.py').read_bytes()).hexdigest()}
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
finally:app.close()
