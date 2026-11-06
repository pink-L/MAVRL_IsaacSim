"""Compare imported upstream encoder/decoder and recurrence, not a duplicate port."""
import ast,importlib.util,unittest
from types import SimpleNamespace
from collections import namedtuple
from pathlib import Path
import numpy as np
import torch as th
from torch import nn
from typing import Tuple
from gymnasium import spaces
from mavrl_isaac.models import Representation
from mavrl_isaac.training import gae

class SourceParity(unittest.TestCase):
 def test_official_encoder_decoder(self):
  path=Path('upstream/mavrl/mav_baselines/torch/recurrent_ppo/recurrent/rnn_extractor.py')
  spec=importlib.util.spec_from_file_location('original_extractor',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  obs=spaces.Dict({'image':spaces.Box(0,255,(1,256,256),np.uint8)})
  original=module.Encoder(obs,64);rep=Representation()
  weights=rep.encoder.state_dict();weights['linear.weight']=weights.pop('fc_mu.weight');weights['linear.bias']=weights.pop('fc_mu.bias')
  original.load_state_dict(weights)
  x=th.rand(3,1,256,256)
  th.testing.assert_close(original(x),rep.encoder(x)[0],rtol=0,atol=0)
  decoder=module.Decoder(obs,64);weights=rep.decoder.state_dict()
  weights['fc.weight']=weights.pop('fc1.weight');weights['fc.bias']=weights.pop('fc1.bias');decoder.load_state_dict(weights)
  z=th.randn(3,64);th.testing.assert_close(decoder(z),rep.decoder(z),rtol=0,atol=0)
 def test_official_lstm_episode_reset(self):
  tree=ast.parse(Path('upstream/mavrl/mav_baselines/torch/recurrent_ppo/recurrent/policies.py').read_text())
  cls=next(x for x in tree.body if isinstance(x,ast.ClassDef))
  f=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_process_sequence')
  f.decorator_list=[];namespace={'th':th,'nn':nn,'Tuple':Tuple,'zip_strict':lambda *args:zip(*args,strict=True)}
  exec(compile(ast.Module(body=[f],type_ignores=[]),'official_process_sequence','exec'),namespace)
  rep=Representation();images=np.random.default_rng(2).integers(0,256,(3,1,256,256),dtype=np.uint8)
  h=tuple(th.randn(1,3,256) for _ in range(2));starts=np.array([False,True,False])
  mu=rep.encoder(th.from_numpy(images).float()/255.)[0]
  expected,state=namespace['_process_sequence'](mu,h,th.tensor(starts,dtype=th.float32),rep.lstm)
  actual,actual_state=rep.step(images,h,starts)
  th.testing.assert_close(expected,actual,rtol=2e-5,atol=3e-7)
  for a,b in zip(state,actual_state):th.testing.assert_close(a,b,rtol=2e-5,atol=3e-7)
 def test_gae_terminal_does_not_leak(self):
  r=th.tensor([[1.],[2.],[100.]]);v=th.zeros_like(r);d=th.tensor([[0.],[1.],[1.]])
  a,_=gae(r,v,d,th.tensor([900.]),1.,1.)
  th.testing.assert_close(a,th.tensor([[3.],[2.],[100.]]))
 def test_source_forward_actor_critic_shared_and_reloaded(self):
  tree=ast.parse(Path('upstream/mavrl/mav_baselines/torch/recurrent_ppo/recurrent/policies.py').read_text())
  cls=next(x for x in tree.body if isinstance(x,ast.ClassDef));states_type=namedtuple('RNNStates',['pi','vf'])
  namespace={'th':th,'nn':nn,'Tuple':Tuple,'RNNStates':states_type,'zip_strict':lambda *args:zip(*args,strict=True)}
  for name in ['_process_sequence','forward_rnn']:
   f=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==name);f.decorator_list=[]
   exec(compile(ast.Module(body=[f],type_ignores=[]),'official_'+name,'exec'),namespace)
  images=np.random.default_rng(7).integers(0,256,(3,1,256,256),dtype=np.uint8);state=th.randn(3,7)
  starts=np.array([False,True,False])
  for separate in [False,True]:
   rep=Representation()
   if separate:rep.enable_separate_critic()
   actor_h=tuple(th.randn(1,3,256) for _ in range(2));critic_h=tuple(th.randn(1,3,256) for _ in range(2))
   original=SimpleNamespace(extract_features=lambda x:rep.encoder(x.float()/255.)[0],share_features_extractor=True,
      _process_sequence=namespace['_process_sequence'],lstm_actor=rep.lstm,lstm_critic=rep.critic_lstm,shared_lstm=not separate)
   pi,vf,expected_states=namespace['forward_rnn'](original,{'image':th.from_numpy(images),'state':state[:,None]},states_type(actor_h,critic_h),th.tensor(starts,dtype=th.float32))
   api,avf,ah,ch=rep.step_both(images,actor_h,critic_h,starts)
   th.testing.assert_close(pi,th.cat([api,state],1),rtol=2e-5,atol=3e-7)
   th.testing.assert_close(vf,th.cat([avf,state],1),rtol=2e-5,atol=3e-7)
   for actual,expected in zip(ah+ch,expected_states.pi+expected_states.vf):th.testing.assert_close(actual,expected,rtol=2e-5,atol=3e-7)
   if separate:self.assertFalse(th.allclose(api,avf))
if __name__=='__main__':unittest.main()
