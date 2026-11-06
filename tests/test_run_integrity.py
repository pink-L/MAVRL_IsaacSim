"""Regression checks for audit findings; no simulator or training required."""
import argparse,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from scripts.run import resolved_config,record_run
from mavrl_isaac.dataset import collect
from mavrl_isaac.scenes import make_scene

class RunIntegrity(unittest.TestCase):
 def setUp(self):
  self.cfg=json.loads(Path('configs/isaac_mavrl.json').read_text())
 def args(self,**kwargs):
  values=dict(resume=None,phase=None,config=None,seed=None,num_envs=None,steps=None,n_steps=None,batch_size=None,warmup_steps=None,schedule_steps=None,epochs=None,current_only=False)
  return argparse.Namespace(**(values|kwargs))
 def test_resume_preserves_final_phase_config_and_schedule(self):
  with tempfile.TemporaryDirectory(dir='.runtime') as directory:
   path=Path(directory)/'policy.pt';cfg=copy.deepcopy(self.cfg);cfg['seed']=17
   torch.save({'phase':'final','config':cfg},path)
   args=self.args(resume=str(path),steps=9000000)
   result=resolved_config(args)
   self.assertEqual(args.phase,'final');self.assertEqual(result['seed'],17)
   self.assertEqual(result['ppo']['total_timesteps'],9000000)
   self.assertEqual(result['ppo']['schedule_timesteps'],8000000)
   with self.assertRaises(ValueError):resolved_config(self.args(resume=str(path),phase='initial'))
 def test_early_stop_does_not_shorten_learning_rate_schedule(self):
  cfg=resolved_config(self.args(steps=8192))
  self.assertEqual(cfg['ppo']['total_timesteps'],8192)
  self.assertEqual(cfg['ppo']['schedule_timesteps'],8000000)
  self.assertEqual(cfg['lstm']['learning_rate'],1e-4)
 def test_third_mission_source_y_interval(self):
  count=0
  for seed in range(40):
   scene=make_scene(seed,self.cfg['scene'],.1,'empty')
   if scene.yaw==np.pi/2:
    count+=1
    self.assertTrue(0<=scene.start[1]<=34 and 0<=scene.goal[1]<=34)
  self.assertGreater(count,0)
 def test_collector_uses_mean_action_and_keeps_episode_splits(self):
  calls=[]
  class Rep:
   def freeze(self):pass
   def zero_state(self,n):return None
   def step(self,image,h,starts):return torch.zeros((2,256)),None
  class Actor:
   def eval(self):pass
   def act(self,features,deterministic=False):
    calls.append(deterministic);return torch.ones((2,4))*float(deterministic),None,None
  class Env:
   n=2
   cfg={'simulation':{}}
   scenes=[SimpleNamespace(seed=5),SimpleNamespace(seed=6)]
   def reset(self):return {'image':np.zeros((2,1,256,256),np.uint8),'state':np.zeros((2,7),np.float32)}
   def step(self,actions):
    np.testing.assert_equal(actions,np.ones((2,4)))
    return self.reset(),None,np.zeros(2,bool),None
  cfg=copy.deepcopy(self.cfg);cfg['device']='cpu'
  with tempfile.TemporaryDirectory(dir='.runtime') as directory:
   policy=Path(directory)/'policy.pt';policy.write_bytes(b'fixture')
   with patch('mavrl_isaac.dataset.load_checkpoint',return_value=(Rep(),Actor(),{})):
    collect(Env(),policy,cfg,Path(directory)/'dataset',24)
   manifest=json.loads((Path(directory)/'dataset/manifest.json').read_text())
   self.assertTrue(manifest['deterministic_policy']);self.assertTrue(all(calls))
   self.assertEqual({f['path'].split('/')[0] for f in manifest['files']},{'train','validation'})
 def test_run_snapshot_hashes_match_copied_sources(self):
  import hashlib
  with tempfile.TemporaryDirectory(dir='.runtime') as directory:
   cfg=copy.deepcopy(self.cfg);record_run(Path(directory),cfg,self.args())
   root=Path(cfg['_run']['source_snapshot'])
   for file,digest in cfg['_run']['source_sha256'].items():
    self.assertEqual(hashlib.sha256((root/file).read_bytes()).hexdigest(),digest)

if __name__=='__main__':unittest.main()
