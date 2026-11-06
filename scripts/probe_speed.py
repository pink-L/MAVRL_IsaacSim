"""Controlled network-input probe, NOT an Isaac navigation evaluation.

Hold all non-speed state values fixed. Feed uniform depth repeatedly to settle
LSTM state, sweep forward speeds, and find positive-to-negative mean ax roots.
No trajectory, success rate, or physical performance is inferred from this.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from mavrl_isaac.training import load_checkpoint,digest
p=argparse.ArgumentParser();p.add_argument('--policy',required=True);p.add_argument('--out',required=True);a=p.parse_args()
rep,actor,_=load_checkpoint(a.policy,'cpu');rep.freeze();actor.eval();torch.set_num_threads(4)
grays=np.arange(40,221,20);velocities=np.linspace(0,8,161)
images=np.broadcast_to(grays[:,None,None,None],(len(grays),1,256,256)).astype(np.uint8).copy()
with torch.no_grad():
 mu=rep.encoder(torch.tensor(images).float()/255.)[0]
 latent,_=rep.lstm(mu[None].expand(100,-1,-1));latent=latent[-1]
 curves=[]
 for i,gray in enumerate(grays):
  state=torch.zeros(len(velocities),7);state[:,0]=np.log(31);state[:,1]=torch.tensor(velocities,dtype=torch.float32)
  x=torch.cat([latent[i].expand(len(velocities),-1),state],1)
  normalized=actor.act(x,True)[0][:,0].numpy();ax=4*normalized
  roots=[]
  for j in range(len(velocities)-1):
   if ax[j]>=0 and ax[j+1]<=0:
    roots.append(float(velocities[j]+(velocities[j+1]-velocities[j])*ax[j]/max(1e-12,ax[j]-ax[j+1])))
  curves.append({'gray':int(gray),'uniform_depth_m':float(gray/255*12),'forward_accel':ax.tolist(),'stable_zero_crossings_mps':roots})
result={'evaluation_type':'controlled_network_input_diagnostic','policy':a.policy,'sha256':digest(a.policy),
 'fixed_state':'[log(31), vx, 0, 0, 0, 0, 0]; LSTM sees 100 identical uniform images; not a physical flight simulation',
 'velocity_grid_mps':velocities.tolist(),'curves':curves}
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
print(json.dumps([{'gray':x['gray'],'stable_zero_crossings_mps':x['stable_zero_crossings_mps']} for x in curves]))
