"""Validate saved artifacts and held-out splits; no claim of trained competence."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from mavrl_isaac.training import load_checkpoint
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root)
manifest=json.loads((root/'dataset/manifest.json').read_text())
sets={s:{f['scene_seed'] for f in manifest['files'] if f['path'].startswith(s+'/')} for s in ['train','validation']}
assert sets['train'] and sets['validation'] and not sets['train']&sets['validation']
rep,actor,checkpoint=load_checkpoint(root/'final/latest.pt','cpu')
representation=torch.load(root/'lstm/best.pt',map_location='cpu',weights_only=False)['representation']
assert checkpoint['representation'].keys()==representation.keys()
assert all(torch.equal(value,representation[key]) for key,value in checkpoint['representation'].items())
rep2,actor2,_=load_checkpoint(root/'final/latest.pt','cpu')
rng=np.random.default_rng(27);image=rng.integers(0,256,(2,1,256,256),dtype=np.uint8)
with torch.no_grad():
 z,h=rep.step(image,rep.zero_state(2),[True,True]);z2,h2=rep2.step(image,rep2.zero_state(2),[True,True])
 x=torch.cat([z,torch.zeros(2,7)],1);x2=torch.cat([z2,torch.zeros(2,7)],1)
 torch.testing.assert_close(actor.act(x,True)[0],actor2.act(x2,True)[0],rtol=0,atol=0)
assert all(torch.isfinite(x).all() for x in checkpoint['actor'].values())
for stage in ['initial','vae','lstm','final']:
 for line in (root/stage/'train.jsonl').read_text().splitlines():
  record=json.loads(line)
  assert all(np.isfinite(v) for v in record.values() if isinstance(v,(int,float)))
evaluation_dir=next((root/name for name in ['evaluation_full','evaluation_sync','evaluation'] if (root/name/'summary.json').exists()),root/'evaluation')
unseen=None
if (evaluation_dir/'episodes.json').exists():
 evaluated={(row['density'],row['scene_seed']) for row in json.loads((evaluation_dir/'episodes.json').read_text())}
 trained=set()
 for stage in ['initial','final']:
  path=root/stage/'episodes.jsonl'
  if path.exists():
   for line in path.read_text().splitlines():
    row=json.loads(line);trained.add((row['density'],row['scene_seed']))
 assert not evaluated&trained,'Evaluation task overlaps a completed training episode'
 unseen=True
result={'checkpoint_load_action_identity':True,'train_validation_scene_disjoint':True,'finite_losses':True,
        'representation_exactly_frozen':True,'separate_critic_present':rep.critic_lstm is not None,
        'evaluation_disjoint_from_logged_training_tasks':unseen,
        'train_scenes':len(sets['train']),'validation_scenes':len(sets['validation']),
        'evaluation_available':(evaluation_dir/'summary.json').exists()}
(root/'validation.json').write_text(json.dumps(result,indent=2));print(result)
