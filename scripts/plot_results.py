"""General research diagnostics. Raw points, no smoothing or convergence claims."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--evaluation-dir',default='evaluation');p.add_argument('--output',default='figures')
a=p.parse_args();root=Path(a.root);out=root/a.output;out.mkdir(exist_ok=True)
def rows(stage):return [json.loads(x) for x in (root/stage/'train.jsonl').read_text().splitlines()]
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
with plt.rc_context({'axes.spines.top':False,'axes.spines.right':False}):
 fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
 for stage,color,marker in [('initial','#0072B2','o'),('final','#D55E00','s')]:
  r=rows(stage)
  axes[0,0].plot([x['timesteps'] for x in r],[x['mean_step_reward'] for x in r],marker=marker,color=color,label=stage)
  axes[0,1].plot([x['timesteps'] for x in r],[np.nan if x['success_rate'] is None else x['success_rate'] for x in r],marker=marker,color=color,label=stage)
 for ax in axes[0]:ax.set_xlabel('Environment transitions');ax.legend()
 axes[0,0].set_ylabel('Mean step reward (raw, phase-dependent)')
 axes[0,1].set(ylabel='Training episode arrival fraction',ylim=(-.02,1.02))
 r=rows('vae')
 axes[1,0].plot([x['epoch'] for x in r],[x['validation_mae_gray'] for x in r],'-o',color='#0072B2')
 axes[1,0].set(xlabel='VAE epoch',ylabel='Held-out MAE (0–255 gray units)')
 r=rows('lstm')
 for key,marker in [('past','o'),('current','s')]:
  if 'validation_'+key+'_mse' in r[0]:axes[1,1].plot([x['epoch'] for x in r],[x['validation_'+key+'_mse'] for x in r],marker=marker,label=key)
 axes[1,1].set(xlabel='LSTM epoch',ylabel='Held-out reconstruction MSE (0–1 depth)');axes[1,1].legend()
 fig.suptitle('MAVRL / Isaac Sim: training diagnostics (not evidence of convergence)')
 fig.savefig(out/'training.png',dpi=180);fig.savefig(out/'training.pdf');plt.close(fig)
 report=json.loads((root/a.evaluation_dir/'summary.json').read_text());results=report['results'];densities=list(results)
 fig,ax=plt.subplots(figsize=(7,4),layout='constrained');x=np.arange(len(densities));w=.23
 for j,(key,title,color) in enumerate([('success_rate','Arrival','#0072B2'),('collision_rate','Collision','#D55E00'),('boundary_rate','Boundary','#009E73')]):
  vals=np.array([results[d][key] for d in densities]);ax.bar(x+(j-1)*w,vals,w,label=title,color=color)
  if key=='success_rate':
   ci=np.array([results[d]['success_95ci'] for d in densities]).T
   errors=np.array([vals-ci[0],ci[1]-vals])
   # At 100/100 Wilson's upper endpoint can round to 1 - 1e-16.
   # Clamp only roundoff; materially inconsistent intervals remain an error.
   if np.any(errors < -1e-12):raise ValueError('Arrival confidence interval excludes its point estimate')
   ax.errorbar(x+(j-1)*w,vals,yerr=np.maximum(errors,0),fmt='none',ecolor='black',capsize=3)
 ax.set(xticks=x,xticklabels=[f"{d}\nn={results[d]['episodes']}" for d in densities],ylabel='Episode fraction',ylim=(0,1.05),title='Held-out tasks; arrival intervals: 95% Wilson')
 ax.legend();fig.savefig(out/'evaluation.png',dpi=180);fig.savefig(out/'evaluation.pdf');plt.close(fig)
 sources=[root/s/'train.jsonl' for s in ['initial','vae','lstm','final']]+[root/a.evaluation_dir/'summary.json']
 manifest={'sources':{str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},'transformations':'No smoothing; raw update/epoch means. Undefined success is NaN. Arrival CI is 95% Wilson over independent scene seeds. Epoch 0 denotes pretraining evaluation. PPO reward schedules differ across phases; do not compare reward magnitudes as success.','alt_text':'Training diagnostics, reconstruction errors and held-out navigation outcomes. See source JSON for all values.','destination':'Internal research report; no journal-specific formatting claim','software':{'matplotlib':matplotlib.__version__,'numpy':np.__version__}}
 (out/'provenance.json').write_text(json.dumps(manifest,indent=2))
