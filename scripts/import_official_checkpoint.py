"""Convert a trusted official release checkpoint without importing ROS/Flightgym.

This is a zero-shot transfer baseline, never a checkpoint trained in Isaac Sim.
Only the legacy Encoder class reference is remapped; all weights load strictly.
"""
import argparse,importlib.util,json,pickle,types
from pathlib import Path
import torch
from mavrl_isaac.models import Representation,ActorCritic
from mavrl_isaac.training import save_checkpoint,digest
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--out',required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('official_rnn_extractor',Path('upstream/mavrl/mav_baselines/torch/recurrent_ppo/recurrent/rnn_extractor.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class LegacyUnpickler(pickle.Unpickler):
    def find_class(self,module_name,name):
        if module_name=='rpg_baselines_prev.torch.recurrent_ppo.recurrent.rnn_extractor' and name=='Encoder':return module.Encoder
        return super().find_class(module_name,name)
compat=types.ModuleType('mavrl_pickle_compat');compat.Unpickler=LegacyUnpickler;compat.load=pickle.load
checkpoint=torch.load(a.source,map_location='cpu',weights_only=False,pickle_module=compat)
s=checkpoint['state_dict'];print('SOURCE_KEYS',list(s),flush=True)
rep=Representation();actor=ActorCritic()
enc={k.removeprefix('features_extractor.'):v for k,v in s.items() if k.startswith('features_extractor.')}
enc['fc_mu.weight']=enc.pop('linear.weight');enc['fc_mu.bias']=enc.pop('linear.bias');rep.encoder.load_state_dict(enc,strict=True)
rep.lstm.load_state_dict({k.removeprefix('lstm_actor.'):v for k,v in s.items() if k.startswith('lstm_actor.')},strict=True)
if any(k.startswith('lstm_critic.') for k in s):
    rep.enable_separate_critic()
    rep.critic_lstm.load_state_dict({k.removeprefix('lstm_critic.'):v for k,v in s.items() if k.startswith('lstm_critic.')},strict=True)
rep.mu_linear.load_state_dict({k.removeprefix('mu_linear.'):v for k,v in s.items() if k.startswith('mu_linear.')},strict=True)
dec={k.removeprefix('feature_decoder0.'):v for k,v in s.items() if k.startswith('feature_decoder0.')}
dec['fc1.weight']=dec.pop('fc.weight');dec['fc1.bias']=dec.pop('fc.bias');rep.decoder.load_state_dict(dec,strict=True)
weights={}
for k,v in s.items():
    if k.startswith('mlp_extractor.policy_net.'):weights['actor.'+k.removeprefix('mlp_extractor.policy_net.')]=v
    elif k.startswith('mlp_extractor.value_net.'):weights['critic.'+k.removeprefix('mlp_extractor.value_net.')]=v
    elif k.startswith(('action_net.','value_net.')) or k=='log_std':weights[k]=v
actor.load_state_dict(weights,strict=True)
Path(a.out).parent.mkdir(parents=True,exist_ok=True)
cfg=json.loads(Path('configs/isaac_mavrl.json').read_text())
save_checkpoint(a.out,rep,actor,cfg,phase='official_zero_shot',source=str(a.source),source_sha256=digest(a.source),timesteps=None)
print('CONVERTED',a.out)
