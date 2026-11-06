"""Sample static scene distributions without simulator or policy, saving geometry."""
import argparse,copy,json,hashlib,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mavrl_isaac.scenes import make_scene,clearance,feasible

p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=30);p.add_argument('--out',default='artifacts/scene_audit');a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cfg=json.loads(Path('configs/isaac_mavrl.json').read_text());rows=[];t0=time.perf_counter()
for distribution in ['uniform','poisson','clustered']:
 for density in ['sparse','medium','dense']:
  sc=copy.deepcopy(cfg['scene']);sc['distribution']=distribution
  for seed in range(2000,2000+a.seeds):
   scene=make_scene(seed,sc,.1,density)
   assert feasible(scene.start,scene.goal,scene.cylinders,40,40,.5,.1)
   margin=float(clearance(np.stack([scene.start,scene.goal]),scene.cylinders).min())
   assert margin>=sc['endpoint_clearance']
   radii=scene.cylinders[:,2]
   rows.append({'seed':seed,'distribution':distribution,'density':density,'obstacles':len(radii),'radius_min':float(radii.min()),'radius_max':float(radii.max()),'endpoint_min_clearance':margin,'traversability':scene.traversability})
summary=[]
for distribution in ['uniform','poisson','clustered']:
 for density in ['sparse','medium','dense']:
  group=[r for r in rows if r['distribution']==distribution and r['density']==density]
  summary.append({'distribution':distribution,'density':density,'tasks':len(group),'mean_obstacles':float(np.mean([r['obstacles'] for r in group])),'min_endpoint_clearance':min(r['endpoint_min_clearance'] for r in group)})
fig,axes=plt.subplots(1,3,figsize=(12,5.2),layout='constrained');scenes=[]
for ax,density in zip(axes,['sparse','medium','dense']):
 scene=make_scene(2026,cfg['scene'],.1,density)
 scenes.append({'seed':scene.seed,'density':density,'cylinders':scene.cylinders.tolist(),'start':scene.start.tolist(),'goal':scene.goal.tolist()})
 for x,y,r in scene.cylinders:ax.add_patch(Circle((x,y),r,facecolor='#999999',edgecolor='#555555',linewidth=.4))
 ax.plot(*scene.start[:2],'o',color='#0072B2',label='Start');ax.plot(*scene.goal[:2],'*',color='#D55E00',markersize=10,label='Goal')
 ax.set(xlim=(-22,22),ylim=(-2,45),aspect='equal',title=f'{density}: {len(scene.cylinders)} cylinders',xlabel='World x (m)',ylabel='World y (m)');ax.legend(fontsize=8)
fig.suptitle('Static, full-height cylinders; feasible tasks; no policy trajectories')
fig.savefig(out/'scenes.png',dpi=180,bbox_inches='tight');fig.savefig(out/'scenes.pdf',bbox_inches='tight');plt.close(fig)
report={'config':cfg,'tasks':len(rows),'all_generated_tasks_pass_geometry_checks':True,'elapsed_s':time.perf_counter()-t0,'summary':summary,'rows':rows,'figure_scenes':scenes,
 'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path('mavrl_isaac/scenes.py'),Path('configs/isaac_mavrl.json')]},
 'scope':'Geometric connectivity and endpoint clearance; no dynamic feasibility, policy competence or shortest-path claim.'}
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2))
