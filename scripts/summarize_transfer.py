"""Report saved official-weight transfer separately from Isaac-trained policies."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('artifacts/official');out=root/'figures';out.mkdir(exist_ok=True)
def read(path):return json.loads(path.read_text())
normal=read(root/'evaluation/summary.json');reset=read(root/'evaluation_reset/summary.json')
replay=read(root/'evaluation_replay/summary.json');assert normal['results']==replay['results']
for filename in ['episodes.json','trajectories.json']:
 assert read(root/'evaluation'/filename)==read(root/'evaluation_replay'/filename)
normal_ep=read(root/'evaluation/episodes.json');reset_ep=read(root/'evaluation_reset/episodes.json')
pairs=[]
for density in normal['results']:
 left={e['scene_seed']:e for e in normal_ep if e['density']==density}
 right={e['scene_seed']:e for e in reset_ep if e['density']==density};assert left.keys()==right.keys()
 counts={'both_success':0,'normal_only_success':0,'reset_only_success':0,'both_fail':0}
 for seed in left:
  a=left[seed]['reason']=='success';b=right[seed]['reason']=='success'
  counts['both_success' if a and b else 'normal_only_success' if a else 'reset_only_success' if b else 'both_fail']+=1
 pairs.append({'density':density,**counts})
probe=read(root/'speed_probe.json');densities=list(normal['results']);x=np.arange(3)
plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for offset,result,label,color in [(-.16,normal,'Normal memory','#0072B2'),(.16,reset,'Reset each frame','#D55E00')]:
 vals=np.array([result['results'][d]['success_rate'] for d in densities]);ci=np.array([result['results'][d]['success_95ci'] for d in densities]).T
 axes[0].bar(x+offset,vals,.3,label=label,color=color)
 axes[0].errorbar(x+offset,vals,yerr=np.maximum(0,[vals-ci[0],ci[1]-vals]),fmt='none',ecolor='black',capsize=3)
 axes[1].plot(x,[result['results'][d]['mean_speed_all_episodes'] for d in densities],'-o',label=label,color=color)
axes[0].set(ylabel='Arrival fraction; 95% Wilson CI',ylim=(0,1.05));axes[0].legend(fontsize=8)
axes[1].set(ylabel='Episode mean horizontal speed (m/s)')
for ax in axes[:2]:ax.set_xticks(x,densities);ax.set_xlabel('Static cylinder density; 20 tasks per density')
curves=[c for c in probe['curves'] if c['stable_zero_crossings_mps']]
axes[2].plot([c['uniform_depth_m'] for c in curves],[c['stable_zero_crossings_mps'][0] for c in curves],'-o',color='#009E73')
axes[2].set(xlabel='Uniform input depth (m)',ylabel='Forward acceleration zero crossing (m/s)',title='Controlled input diagnostic; not flight')
fig.suptitle('Official pretrained MAVRL: zero-shot Isaac transfer and diagnostics')
fig.savefig(out/'transfer.png',dpi=180);fig.savefig(out/'transfer.pdf');plt.close(fig)
files=[root/f'{stage}/{name}' for stage in ['evaluation','evaluation_reset','evaluation_replay'] for name in ['summary.json','episodes.json','trajectories.json']]+[root/'speed_probe.json']
report={'exact_replay':True,'paired_outcomes':pairs,'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
 'scope':'Official pretrained actor, not trained in Isaac. One weight, 20 seeds per density. Inference-time memory reset is not a retrained ablation. All-episode speed is confounded by failed episodes. Uniform-depth input probe is not simulated navigation.',
 'historical_provenance':'These captures predate per-run source snapshots. Original scene area-2 y interval was [0,40], corrected later to source [0,34]. Exact replay validates these saved legacy runs only.',
 'software':{'matplotlib':matplotlib.__version__,'numpy':np.__version__}}
(out/'provenance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['paired_outcomes'],indent=2))
