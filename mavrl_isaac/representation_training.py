import json,copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from torchvision.utils import save_image
from .models import VAE,Representation,ActorCritic
from .training import seed_all,load_checkpoint,save_checkpoint,digest
from .dataset import load_sequences


def vae_loss(recon,x,mu,logsigma):
    return F.mse_loss(recon,x,reduction='sum')-.5*torch.sum(1+2*logsigma-mu.square()-(2*logsigma).exp())


def train_vae(cfg,dataset,out):
    seed_all(cfg['seed']);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    device=cfg['device'];v=cfg['vae'];model=VAE(1,64).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=v['learning_rate'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=5)
    loaders={s:DataLoader(TensorDataset(torch.from_numpy(np.concatenate(load_sequences(dataset,s)))),batch_size=v['batch_size'],shuffle=s=='train') for s in ['train','validation']}
    best=float('inf');stale=0;log=open(out/'train.jsonl','w',buffering=1)
    for epoch in range(v['epochs']+1):
        metrics={}
        for split,loader in loaders.items():
            training=split=='train' and epoch>0
            model.train(training);total=mse=kl=mae=0.;pixels=n=0
            for (image,) in loader:
                x=image.to(device).float()/255.
                if training:
                    mask=torch.rand(len(x),device=device)<.5
                    x[mask]=x[mask].flip(-1)
                with torch.set_grad_enabled(training):
                    recon,mu,logsigma=model(x);loss=vae_loss(recon,x,mu,logsigma)
                    if training:optimizer.zero_grad();loss.backward();optimizer.step()
                total+=loss.item();mse+=F.mse_loss(recon,x,reduction='sum').item()
                mae+=(recon-x).abs().sum().item();pixels+=x.numel();n+=len(x)
            metrics[split+'_loss_per_image']=total/n;metrics[split+'_mse']=mse/pixels;metrics[split+'_mae_gray']=mae/pixels*255
        metrics['epoch']=epoch;log.write(json.dumps(metrics)+'\n');print('VAE',json.dumps(metrics),flush=True)
        score=metrics['validation_loss_per_image'];scheduler.step(score)
        state={'vae':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'config':cfg,'epoch':epoch,'metrics':metrics,'dataset_sha256':digest(Path(dataset)/'manifest.json')}
        torch.save(state,out/'latest.pt')
        if score<best:best=score;stale=0;torch.save(state,out/'best.pt')
        else:stale+=1
        if epoch==0 or epoch==v['epochs']:
            save_image(torch.cat([x[:8],recon[:8]],0),out/f'reconstruction_{epoch:04d}.png',nrow=8)
        if stale>=v['early_stop_patience']:break
    log.close()


def reconstruction_loss(rep,mu,images,lag,members,reset_every_frame=False):
    if reset_every_frame:
        latent,_=rep.lstm(mu[None])  # batch is time, single independent LSTM step
        latent=latent[0]
    else:
        latent,_=rep.lstm(mu[:,None]);latent=latent[:,0]
    projections=rep.mu_linear(latent).split(64,-1)
    loss=0;metrics={};outputs=[]
    for j,enabled in enumerate(members):
        if not enabled:outputs.append(None);continue
        z=projections[j][lag:] if j==0 else projections[j][:-lag] if j==2 else projections[j]
        target=images[:-lag] if j==0 else images[lag:] if j==2 else images
        recon=rep.decoder(z);loss=loss+F.mse_loss(recon,target,reduction='sum')
        metrics[['past','current','future'][j]+'_mse']=F.mse_loss(recon,target).item()
        outputs.append(recon)
    return loss,metrics,outputs


def train_lstm(cfg,dataset,vae_path,initial_policy,out):
    seed_all(cfg['seed']);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    device=cfg['device'];lc=cfg['lstm'];rep,actor,_=load_checkpoint(initial_policy,device)
    rep.enable_separate_critic()
    vae=torch.load(vae_path,map_location=device,weights_only=False)
    encoder={k.removeprefix('encoder.'):v for k,v in vae['vae'].items() if k.startswith('encoder.')}
    rep.encoder.load_state_dict(encoder)
    rep.requires_grad_(True);rep.encoder.requires_grad_(False)
    # Upstream's offline reconstruction only traverses actor LSTM; the critic
    # LSTM gets no gradients and stays at its reload-time initialization.
    rep.critic_lstm.requires_grad_(False)
    opt=torch.optim.Adam([p for p in rep.parameters() if p.requires_grad],lr=lc['learning_rate'],eps=1e-5)
    datasets={}
    for split in ['train','validation']:
        datasets[split]=[]
        for array in load_sequences(dataset,split):
            x=torch.from_numpy(array).to(device).float()/255.
            with torch.no_grad():mu,_=rep.encoder(x)
            datasets[split].append((mu.cpu(),torch.from_numpy(array)))
    best=float('inf');log=open(out/'train.jsonl','w',buffering=1)
    for epoch in range(lc['epochs']+1):
        stats={}
        for split,sequences in datasets.items():
            train=split=='train' and epoch>0;rep.train(train)
            indices=np.random.permutation(len(sequences)) if train else range(len(sequences))
            metrics=[]
            for idx in indices:
                mu,raw=sequences[idx];mu=mu.to(device);x=raw.to(device).float()/255.
                with torch.set_grad_enabled(train):
                    loss,record,outputs=reconstruction_loss(rep,mu,x,lc['reconstruction_steps'],lc['reconstruction_members'])
                    if train:opt.zero_grad();loss.backward();opt.step()
                metrics.append(record)
            for key in metrics[0]:stats[split+'_'+key]=float(np.mean([r[key] for r in metrics]))
        stats['epoch']=epoch;log.write(json.dumps(stats)+'\n');print('LSTM',json.dumps(stats),flush=True)
        score=sum(v for k,v in stats.items() if k.startswith('validation_'))
        extra={'phase':'representation','epoch':epoch,'metrics':stats,'vae_sha256':digest(vae_path),'dataset_sha256':digest(Path(dataset)/'manifest.json')}
        save_checkpoint(out/'latest.pt',rep,actor,cfg,opt,**extra)
        if score<best:best=score;save_checkpoint(out/'best.pt',rep,actor,cfg,opt,**extra)
        if epoch in (0,lc['epochs']):
            lag=lc['reconstruction_steps'];pieces=[x[:8],x[lag:lag+8]]
            if outputs[0] is not None:pieces.append(outputs[0][:8])
            if outputs[1] is not None:pieces.append(outputs[1][lag:lag+8])
            save_image(torch.cat(pieces),out/f'reconstruction_{epoch:04d}.png',nrow=8)
    log.close()
    rep,_,_=load_checkpoint(out/'best.pt',device)
    results=[]
    for mu,raw in datasets['validation']:
        with torch.no_grad():
            _,normal,_=reconstruction_loss(rep,mu.to(device),raw.to(device).float()/255.,lc['reconstruction_steps'],lc['reconstruction_members'])
            _,reset,_=reconstruction_loss(rep,mu.to(device),raw.to(device).float()/255.,lc['reconstruction_steps'],lc['reconstruction_members'],True)
        results.append({'normal':normal,'reset_memory':reset})
    (out/'memory_diagnostic.json').write_text(json.dumps({'per_sequence':results,'interpretation':'Inference-time hidden-state reset is a diagnostic, not a separately retrained no-memory baseline.'},indent=2))
