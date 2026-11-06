"""Serial continuation and evaluation of an already-validated short pipeline.

Designed for a long single-GPU run. Records live stage/PID/command and stops on
errors or another GPU compute job before beginning a new simulator stage.
"""
import argparse,json,os,subprocess,sys,time,shutil,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--steps',type=int,default=8000000);p.add_argument('--episodes',type=int,default=100)
p.add_argument('--formal',action='store_true');p.add_argument('--initial-policy');p.add_argument('--frames',type=int,default=65536)
p.add_argument('--num-envs',type=int,default=16,help='Final PPO parallel environments; collection/evaluation stay at 16')
p.add_argument('--adopt-lstm-pid',type=int,help='Adopt this project\'s already running LSTM after replacing only its supervisor')
p.add_argument('--reuse-inputs-from',help='Reuse initial policy, dataset and VAE from a completed baseline for a paired representation ablation')
p.add_argument('--current-only',action='store_true',help='Train the LSTM with reconstruction members [0,1,0]')
p.add_argument('--config',help='Explicit simulator/training configuration, recorded by every run.py stage')
p.add_argument('--prepared-root',action='store_true',help='Use completed initial/data/VAE/LSTM artifacts already linked into root; train a fresh final PPO')
p.add_argument('--postprocess-only',action='store_true',help='Finish plots/report from validated existing evaluations; do not retrain or reevaluate')
a=p.parse_args()
if a.num_envs<1:p.error('--num-envs must be positive')
if a.adopt_lstm_pid and not a.formal:p.error('--adopt-lstm-pid requires --formal')
if a.reuse_inputs_from and (not a.formal or a.adopt_lstm_pid):p.error('--reuse-inputs-from requires --formal and cannot adopt a live LSTM')
if a.current_only and not a.formal:p.error('--current-only requires --formal; it cannot change an existing PPO checkpoint')
if a.prepared_root and (not a.formal or a.adopt_lstm_pid or a.reuse_inputs_from):p.error('--prepared-root requires --formal and cannot adopt or reuse inputs through another mode')
root=Path(a.root);folder=root/'long_run';folder.mkdir(parents=True,exist_ok=True)
status={'pid':os.getpid(),'formal_representation_budget':a.formal,'requested_steps':a.steps,'ppo_num_envs':a.num_envs,'evaluation_episodes_per_density':a.episodes,'root':str(root),'current_only':a.current_only,'reuse_inputs_from':a.reuse_inputs_from,'prepared_root':a.prepared_root,'config':a.config,'state':'starting'}
def update(**values):
 status.update(values);status['updated_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
 (folder/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
def gpu_available():
 rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
 if rows:raise RuntimeError('Another GPU compute process is present; continuation stopped before starting the next GPU stage: '+rows)
def run(stage,args,gpu=False):
 if gpu:gpu_available()
 if a.config and args[0]=='scripts/run.py':args=[*args,'--config',a.config]
 command=['scripts/python.sh',*args];update(state='running',stage=stage,command=command,child_pid=None)
 with (folder/f'{stage}.log').open('w') as log:
  process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
  update(child_pid=process.pid);code=process.wait()
 if code:raise RuntimeError(f'{stage} exited with {code}; see {folder/stage}.log')
 update(last_completed_stage=stage,child_pid=None)
try:
 if a.postprocess_only:
  checkpoint_hash=hashlib.sha256((root/'final/latest.pt').read_bytes()).hexdigest()
  for name in ['evaluation_full','evaluation_full_reset','evaluation_full_replay']:
   saved=json.loads((root/name/'summary.json').read_text())
   if saved['sha256']!=checkpoint_hash:raise ValueError('Saved evaluation has a different policy: '+name)
  checks=json.loads((root/'validation.json').read_text())
  if not all(checks.get(k) for k in ['checkpoint_load_action_identity','representation_exactly_frozen','finite_losses','separate_critic_present']):
   raise ValueError('Missing successful artifact checks')
  update(last_completed_stage='validation',postprocess_only=True)
 elif a.formal:
  if not a.initial_policy:raise ValueError('--formal requires --initial-policy')
  source=Path(a.initial_policy)
  if a.prepared_root:
   if (root/'final').exists():raise ValueError('Prepared root already has final/; refusing to overwrite PPO')
   if source.resolve()!=(root/'initial/latest.pt').resolve():raise ValueError('Initial policy differs from prepared root')
   import torch
   checkpoint=torch.load(root/'lstm/latest.pt',map_location='cpu',weights_only=False)
   expected=[0,1,0] if a.current_only else [1,1,0]
   if checkpoint['epoch']!=checkpoint['config']['lstm']['epochs'] or checkpoint['config']['lstm']['reconstruction_members']!=expected:
    raise ValueError('Prepared LSTM is incomplete or has the wrong reconstruction objective')
   best=torch.load(root/'lstm/best.pt',map_location='cpu',weights_only=False)
   for name,key in [('vae/best.pt','vae_sha256'),('dataset/manifest.json','dataset_sha256')]:
    if best[key]!=hashlib.sha256((root/name).read_bytes()).hexdigest():raise ValueError('Prepared representation input hash differs: '+name)
   if not json.loads((root/'lstm/memory_diagnostic.json').read_text())['per_sequence']:raise ValueError('Missing LSTM completion diagnostic')
   del checkpoint,best
   update(last_completed_stage='lstm',prepared_completion='Full LSTM epoch budget, objective and input hashes verified')
  elif a.adopt_lstm_pid:
   process=Path(f'/proc/{a.adopt_lstm_pid}')
   if process.exists():
    command=(process/'cmdline').read_text().replace('\0',' ')
    if 'scripts/run.py lstm' not in command or str(root/'lstm') not in command:
     raise ValueError('Adopted PID does not match this run\'s LSTM command')
   update(state='running',stage='lstm',child_pid=a.adopt_lstm_pid,adopted_stage=True,last_completed_stage='vae')
   while process.exists():
    try:
     if (process/'stat').read_text().split(') ',1)[1].split()[0]=='Z':break
    except FileNotFoundError:break
    time.sleep(2)
   # An adopted process has no wait() exit status. Require its final artifacts,
   # including the diagnostic written only after all configured epochs finish.
   import torch
   checkpoint=torch.load(root/'lstm/latest.pt',map_location='cpu',weights_only=False)
   if checkpoint['epoch']!=checkpoint['config']['lstm']['epochs']:
    raise RuntimeError('Adopted LSTM stopped before its configured final epoch')
   diagnostic=json.loads((root/'lstm/memory_diagnostic.json').read_text())
   if not diagnostic['per_sequence']:raise RuntimeError('Missing completed LSTM validation diagnostic')
   del checkpoint
   update(last_completed_stage='lstm',child_pid=None,adopted_completion='Final epoch and post-training diagnostic verified; original exit status unavailable')
  else:
   if (root/'initial').exists():raise ValueError('Formal output already contains initial/; select a new output directory to preserve evidence')
   if a.reuse_inputs_from:
    baseline=Path(a.reuse_inputs_from).resolve()
    if json.loads((baseline/'long_run/status.json').read_text())['state']!='completed':
     raise ValueError('Paired ablation requires the baseline pipeline to have completed its validation and evaluation')
    if source.resolve()!=(baseline/'initial/latest.pt').resolve():
     raise ValueError('Paired ablation must reuse the baseline initial policy')
    inputs={name:str(baseline/name) for name in ['initial','dataset','vae']}
    files=[baseline/'initial/latest.pt',baseline/'dataset/manifest.json',baseline/'vae/best.pt']
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    for name,target in inputs.items():(root/name).symlink_to(target,target_is_directory=True)
    (root/'REUSED_INPUTS.json').write_text(json.dumps({'baseline':str(baseline),'directories':inputs,'sha256':hashes,'scope':'Shared data/VAE/initial policy for a paired reconstruction-objective ablation; not an independent baseline training seed'},indent=2))
   else:
    shutil.copytree(source.parent,root/'initial')
    (root/'initial/REUSED_FROM.json').write_text(json.dumps({'source':str(source.resolve()),'meaning':'Previously trained initial policy reused; this is not another independent initial training seed'},indent=2))
    run('collect',['scripts/run.py','collect','--policy',str(source),'--out',str(root/'dataset'),'--frames',str(a.frames),'--num-envs','16','--seed','100'],True)
    run('vae',['scripts/run.py','vae','--dataset',str(root/'dataset'),'--out',str(root/'vae')],True)
   run('lstm',['scripts/run.py','lstm','--dataset',str(root/'dataset'),'--vae',str(root/'vae/best.pt'),'--policy',str(source),'--out',str(root/'lstm'),*(['--current-only'] if a.current_only else [])],True)
  run('train',['scripts/run.py','train','--phase','final','--representation',str(root/'lstm/best.pt'),'--out',str(root/'final'),'--steps',str(a.steps),'--num-envs',str(a.num_envs),*(['--current-only'] if a.current_only else [])],True)
 else:
  validation=json.loads((root/'validation.json').read_text())
  assert validation['representation_exactly_frozen'] and validation['separate_critic_present']
  run('train',['scripts/run.py','train','--resume',str(root/'final/latest.pt'),'--out',str(root/'final'),'--steps',str(a.steps)],True)
 if not a.postprocess_only:
  for name,extra in [('evaluation_full',[]),('evaluation_full_reset',['--reset-memory']),('evaluation_full_replay',[])]:
   run(name,['scripts/run.py','evaluate','--policy',str(root/'final/latest.pt'),'--out',str(root/name),'--episodes',str(a.episodes),'--num-envs','16','--seed','200',*extra,*(['--current-only'] if a.current_only else [])],True)
 normal=json.loads((root/'evaluation_full/summary.json').read_text());replay=json.loads((root/'evaluation_full_replay/summary.json').read_text())
 assert normal['results']==replay['results'],'Repeated evaluation differs; inspect saved runs'
 if not a.postprocess_only:
  run('speed_probe',['scripts/probe_speed.py','--policy',str(root/'final/latest.pt'),'--out',str(root/'speed_probe.json')])
  run('validation',['scripts/validate_artifacts.py',str(root)])
 run('plots',['scripts/plot_results.py',str(root),'--evaluation-dir','evaluation_full','--output','figures_full'])
 run('memory_comparison',['scripts/plot_memory_comparison.py',str(root),'--normal','evaluation_full','--reset','evaluation_full_reset','--replay','evaluation_full_replay','--output','memory_figures_full'])
 reset=json.loads((root/'evaluation_full_reset/summary.json').read_text())
 scope='Default VAE/LSTM epoch budgets; initial policy reused from the recorded short initial training.' if a.formal else 'Continues the short representation pipeline (VAE20/LSTM30 epochs), not the full official representation budget.'
 if a.reuse_inputs_from:scope=f"Paired reconstruction ablation, members={'[0,1,0]' if a.current_only else '[1,1,0]'}; data, VAE and initial policy reused from {a.reuse_inputs_from}; full LSTM/PPO budgets."
 if a.prepared_root:scope='Fresh PPO with prepared, hash-verified initial/data/VAE/LSTM inputs. Their original source configuration remains recorded; see REUSED_INPUTS.json.'
 if normal['config']['reward'].get('speed_mode')=='target':scope+=f" Target-speed reward ablation: target={normal['config']['reward']['target_speed']} m/s, coefficient={normal['config']['reward']['final'][1]}; project-selected parameters."
 lines=['# Long pipeline run result','',f"Completed at {datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}",'',
  f"PPO transitions: {normal['training_timesteps']}. {scope} One training seed; static Isaac cylinders only.",'',
  '| Density | Episodes | Arrival | Collision | Mean speed (m/s) | Arrival with memory reset |','|---|---:|---:|---:|---:|---:|']
 for density,result in normal['results'].items():
  lines.append(f"| {density} | {result['episodes']} | {result['success_rate']:.3f} | {result['collision_rate']:.3f} | {result['mean_speed_all_episodes']:.3f} | {reset['results'][density]['success_rate']:.3f} |")
 lines+=['','The normal evaluation was repeated with identical summary results. Wilson intervals, seeds, trajectories, configurations and hashes are in the corresponding evaluation directories. Inference-time reset is not a retrained ablation. Additional training seeds were excluded by user request on 2026-11-05; results are limited to one training seed. Mechanism claims require the separately saved paired reconstruction and speed comparisons.']
 (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
 update(state='completed',stage='finished',results=str(folder/'RESULTS.md'))
except BaseException as exc:
 update(state='failed',error=repr(exc));raise
