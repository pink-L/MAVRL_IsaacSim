"""Paired inference-time memory diagnostic and exact saved-policy replay check."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--normal',default='evaluation');p.add_argument('--reset',default='evaluation_reset');p.add_argument('--replay',default='evaluation_replay');p.add_argument('--output',default='memory_figures');a=p.parse_args()
root=Path(a.root);out=root/a.output;out.mkdir(exist_ok=True)
def read(stage,name):return json.loads((root/stage/name).read_text())
n=read(a.normal,'summary.json');r=read(a.reset,'summary.json');repeat=read(a.replay,'summary.json')
assert n['sha256']==r['sha256']==repeat['sha256']
assert n['results']==repeat['results']
for name in ['episodes.json','trajectories.json']:assert read(a.normal,name)==read(a.replay,name)
normal_ep=read(a.normal,'episodes.json');reset_ep=read(a.reset,'episodes.json');pairs=[]
for density in n['results']:
 assert n['results'][density]['seeds']==r['results'][density]['seeds']
 left={e['scene_seed']:e for e in normal_ep if e['density']==density};right={e['scene_seed']:e for e in reset_ep if e['density']==density}
 counts={'both_success':0,'normal_only_success':0,'reset_only_success':0,'both_fail':0}
 for seed in left:
  x=left[seed]['reason']=='success';y=right[seed]['reason']=='success'
  counts['both_success' if x and y else 'normal_only_success' if x else 'reset_only_success' if y else 'both_fail']+=1
 pairs.append({'density':density,**counts})
fig,axes=plt.subplots(1,2,figsize=(9,4.2),layout='constrained');densities=list(n['results']);x=np.arange(len(densities))
for offset,result,label,color in [(-.16,n,'Normal memory','#0072B2'),(.16,r,'Reset every frame','#D55E00')]:
 vals=np.array([result['results'][d]['success_rate'] for d in densities]);ci=np.array([result['results'][d]['success_95ci'] for d in densities]).T
 axes[0].bar(x+offset,vals,.3,color=color,label=label)
 axes[0].errorbar(x+offset,vals,yerr=np.maximum(0,[vals-ci[0],ci[1]-vals]),fmt='none',ecolor='black',capsize=3)
 axes[1].plot(x,[result['results'][d]['mean_speed_all_episodes'] for d in densities],'-o',color=color,label=label)
axes[0].set(ylabel='Arrival fraction; 95% Wilson intervals',ylim=(0,1.05));axes[0].legend(fontsize=8)
axes[1].set(ylabel='All-episode mean horizontal speed (m/s)')
for ax in axes:ax.set_xticks(x,[f"{d}\nn={n['results'][d]['episodes']}" for d in densities]);ax.set_xlabel('Paired static cylinder tasks')
fig.suptitle('Isaac-trained PPO: inference memory diagnostic, not retrained ablation')
fig.savefig(out/'memory.png',dpi=180);fig.savefig(out/'memory.pdf');plt.close(fig)
files=[root/stage/name for stage in [a.normal,a.reset,a.replay] for name in ['summary.json','episodes.json','trajectories.json']]
result={'exact_replay':True,'paired_outcomes':pairs,'sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
 'scope':'One weight, inference-time reset, not a separately retrained current-only baseline. All-episode speed is affected by failures. Exact replay is limited to this hardware/software/capture configuration.'}
(out/'provenance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['paired_outcomes'],indent=2))
