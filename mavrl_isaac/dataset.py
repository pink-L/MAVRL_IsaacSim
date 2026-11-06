import json
from pathlib import Path
import numpy as np
import torch
from .training import load_checkpoint,digest


def collect(env,policy,cfg,out,frames):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rep,actor,_=load_checkpoint(policy,cfg['device']);rep.freeze();actor.eval()
    env.cfg['simulation']['reset_if_collide']=False
    obs=env.reset();h=rep.zero_state(env.n);starts=np.ones(env.n,bool)
    buffers=[[] for _ in range(env.n)];states=[[] for _ in range(env.n)]
    group=[0]*env.n;files=[];counter=0;length=cfg['lstm']['sequence_length']
    def flush(i):
        nonlocal counter
        if len(buffers[i])<=cfg['lstm']['reconstruction_steps']:
            buffers[i]=[];states[i]=[];return
        scene=env.scenes[i]
        split='validation' if group[i]%5==0 else 'train'
        name=f'{split}/sequence_{counter:05d}.npz';(out/split).mkdir(exist_ok=True)
        np.savez_compressed(out/name,image=np.stack(buffers[i]),state=np.stack(states[i]),scene_seed=group[i])
        files.append({'path':name,'frames':len(buffers[i]),'scene_seed':group[i]});counter+=1
        buffers[i]=[];states[i]=[]
    group=[s.seed for s in env.scenes]
    for step in range((frames+env.n-1)//env.n):
        for i in range(env.n):
            buffers[i].append(obs['image'][i]);states[i].append(obs['state'][i])
        with torch.no_grad():
            latent,h=rep.step(obs['image'],h,starts)
            action,_,_=actor.act(torch.cat([latent,torch.as_tensor(obs['state'],device=cfg['device'])],1),deterministic=True)
        # Flush BEFORE environment autoreset so sequence labels stay episode-local.
        obs,_,done,_=env.step(action.cpu().numpy());starts=done
        for i in range(env.n):
            if done[i] or len(buffers[i])>=length:flush(i)
            if done[i]:group[i]=env.scenes[i].seed
        if step%100==0:print('COLLECT',step*env.n,flush=True)
    for i in range(env.n):flush(i)
    manifest={'policy':str(policy),'policy_sha256':digest(policy),'config':cfg,'files':files,'frames':sum(f['frames'] for f in files),
        'split_rule':'scene_seed % 5 == 0 is validation; no scene is shared between splits',
        'collision_termination':False,'deterministic_policy':True,'hidden_state':'zero at contiguous sequence start; no cross-episode targets'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if not any(f['path'].startswith('validation/') for f in files):raise RuntimeError('Collected no validation scenes; collect more frames with another seed')
    print('DATASET',manifest['frames'],len(files),flush=True)


def load_sequences(root,split):
    manifest=json.loads((Path(root)/'manifest.json').read_text())
    files=[f for f in manifest['files'] if f['path'].startswith(split+'/')]
    if not files:raise ValueError(f'No {split} sequences in {root}')
    result=[]
    for f in files:
        with np.load(Path(root)/f['path']) as data:result.append(data['image'].copy())
    return result
