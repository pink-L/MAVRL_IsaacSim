import copy,json,unittest
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from mavrl_isaac.dynamics import integrate,observation,progress_reward,preprocess_depth
from mavrl_isaac.scenes import make_scene,feasible,segment_collision
from mavrl_isaac.environment import NavigationEnv
from mavrl_isaac.models import Representation,ActorCritic,VAE

class CoreTests(unittest.TestCase):
 def setUp(self):
  self.cfg=json.loads(Path('configs/isaac_mavrl.json').read_text());self.cfg['num_envs']=1
 def test_coordinate_and_acceleration(self):
  p,v,r,y=integrate(np.zeros((1,3)),np.array([[2.,0.,0.]]),np.eye(3)[None],np.array([[4.,0.,0.,.6]]),.1)
  np.testing.assert_allclose(p,[[0,.22,0]],atol=1e-12)
  np.testing.assert_allclose(v,[[2.4,0,0]]);self.assertAlmostEqual(y[0],.06)
  p,*_=integrate(np.zeros((1,3)),np.array([[1.,0.,0.]]),Rotation.from_euler('z',[-np.pi/2]).as_matrix(),np.zeros((1,4)),.1)
  np.testing.assert_allclose(p,[[.1,0,0]],atol=1e-12)
 def test_observation_and_reward(self):
  obs=observation(np.array([[0.,0.,3.]]),np.array([[4.,0.,.2]]),np.array([[0.,3.,4.]]),np.zeros(1))
  np.testing.assert_allclose(obs,[[np.log(4),4,0,0,1,.2,0]])
  r,_=progress_reward(obs,np.array([[2.,0,0]]),np.zeros((1,3)),self.cfg['reward']['initial'])
  self.assertAlmostEqual(r[0],-.001*np.log(4)-.004-.002-.001)
 def test_scene_reproducibility_feasibility(self):
  counts=[]
  for density in ['sparse','medium','dense']:
   sizes=[]
   for seed in range(4):
    a=make_scene(seed,self.cfg['scene'],.1,density);b=make_scene(seed,self.cfg['scene'],.1,density)
    np.testing.assert_array_equal(a.cylinders,b.cylinders)
    self.assertTrue(feasible(a.start,a.goal,a.cylinders,40,40,.5,.1));sizes.append(len(a.cylinders))
   counts.append(np.mean(sizes))
  self.assertTrue(counts[0]<counts[1]<counts[2])
 def test_swept_collision(self):
  c=np.array([[0.,0.,1.]])
  self.assertTrue(segment_collision(np.array([-2.,0,3]),np.array([2.,0,3]),c,.1))
  self.assertFalse(segment_collision(np.array([-2.,2,3]),np.array([2.,2,3]),c,.1))
 def test_terminal_and_reset(self):
  env=NavigationEnv(self.cfg,density='empty');env.reset([123]);env.position[0]=env.goal[0]
  _,r,d,info=env.step(np.zeros((1,4)))
  self.assertTrue(d[0]);self.assertEqual(info[0]['episode']['reason'],'success')
  self.assertAlmostEqual(r[0],20/info[0]['episode']['traversability'],places=6)
  self.assertEqual(env.steps[0],0)
 def test_depth_scale(self):
  np.testing.assert_array_equal(preprocess_depth(np.array([0.,6.,12.,15.,np.inf])),[0,127,255,255,255])
 def test_lstm_reset_and_frozen_gradients(self):
  torch.manual_seed(0);rep=Representation().freeze();image=np.zeros((2,1,256,256),dtype=np.uint8)
  h=rep.zero_state(2);first,_=rep.step(image,h,[True,True]);old=tuple(torch.randn_like(s) for s in h)
  reset,_=rep.step(image,old,[True,True]);torch.testing.assert_close(first,reset)
  ac=ActorCritic();dist,val=ac.distribution_value(torch.cat([first,torch.zeros(2,7)],1));loss=val.sum()+dist.mean.sum();loss.backward()
  self.assertTrue(all(p.grad is None for p in rep.parameters()))
  self.assertTrue(any(p.grad is not None for p in ac.parameters()))
 def test_official_vae_shapes(self):
  vae=VAE(1,64);x=torch.rand(2,1,256,256);recon,mu,logsigma=vae(x)
  self.assertEqual(recon.shape,x.shape);self.assertEqual(mu.shape,(2,64));self.assertTrue(torch.isfinite(recon).all())

if __name__=='__main__':unittest.main()
