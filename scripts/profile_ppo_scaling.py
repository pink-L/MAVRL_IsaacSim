"""Real RTX rollout plus source PPO updates; benchmark weights are discarded."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from run import record_run
from mavrl_isaac.training import seed_all,load_checkpoint,gae,ppo_update


def main():
    p=argparse.ArgumentParser();p.add_argument('--num-envs',type=int,required=True)
    p.add_argument('--out',required=True);p.add_argument('--warmup',type=int,default=150)
    p.add_argument('--samples',type=int,default=64000)
    p.add_argument('--policy',default='artifacts/capture2_baseline_seed0/final/latest.pt')
    p.add_argument('--config',default='configs/isaac_capture2.json')
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[4000,8000,16000])
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path(a.config).read_text());cfg['num_envs']=a.num_envs
    steps=int(np.ceil(a.samples/a.num_envs));cfg['ppo']['n_steps']=steps
    record_run(out,cfg,a);seed_all(cfg['seed'])
    started=time.perf_counter()
    def progress(stage,**extra):
        (out/'status.json').write_text(json.dumps({'pid':os.getpid(),'stage':stage,
            'elapsed_s':time.perf_counter()-started,**extra},indent=2)+'\n')
    progress('startup')
    from mavrl_isaac.renderer import launch,IsaacDepthRenderer
    app=launch()
    try:
        from mavrl_isaac import environment as environment_module
        from mavrl_isaac import renderer as renderer_module
        import warp as wp
        from mavrl_isaac.environment import NavigationEnv
        renderer=IsaacDepthRenderer(cfg)
        costs={'pose':0.,'rtx_step':0.,'render_total':0.,'scene_generation':0.,
               'tiled_read':0.,'warp_sync':0.,'depth_preprocess':0.}
        def timed(function,key):
            def wrapped(*args,**kwargs):
                t=time.perf_counter()
                try:return function(*args,**kwargs)
                finally:costs[key]+=time.perf_counter()-t
            return wrapped
        for camera in renderer.cameras:camera.set_world_pose=timed(camera.set_world_pose,'pose')
        renderer.rep.orchestrator.step=timed(renderer.rep.orchestrator.step,'rtx_step')
        renderer.render=timed(renderer.render,'render_total')
        renderer.view.get_depth=timed(renderer.view.get_depth,'tiled_read')
        wp.synchronize_device=timed(wp.synchronize_device,'warp_sync')
        renderer_module.preprocess_depth=timed(renderer_module.preprocess_depth,'depth_preprocess')
        environment_module.make_scene=timed(environment_module.make_scene,'scene_generation')
        env=NavigationEnv(cfg,renderer,reward_phase='final',density='mixed')
        rep,actor,_=load_checkpoint(a.policy,cfg['device']);actor.eval()
        obs=env.reset();h=rep.zero_state(env.n);hv=rep.zero_state(env.n);starts=np.ones(env.n,bool)
        def infer():
            nonlocal h,hv
            with torch.no_grad():
                z,zv,h,hv=rep.step_both(obs['image'],h,hv,starts)
                state=torch.as_tensor(obs['state'],device=cfg['device'])
                f,fv=torch.cat([z,state],-1),torch.cat([zv,state],-1)
                action,value,log=actor.act(f,critic_x=fv)
            return f,fv,action,value,log,action.cpu().numpy()
        progress('warmup',steps=a.warmup)
        for i in range(a.warmup):
            *_,action=infer();obs,_,starts,_=env.step(action)
            if i%25==0:progress('warmup',completed=i,total=a.warmup)
        setup_seconds=time.perf_counter()-started
        for key in costs:costs[key]=0.
        data=[];inference_s=env_s=0.;episodes=0
        torch.cuda.synchronize();t_rollout=time.perf_counter()
        progress('rollout',total=steps)
        for i in range(steps):
            t=time.perf_counter();f,fv,action,value,log,numpy_action=infer();inference_s+=time.perf_counter()-t
            t=time.perf_counter();obs,reward,starts,infos=env.step(numpy_action);env_s+=time.perf_counter()-t
            episodes+=sum('episode' in info for info in infos)
            data.append((f,fv,action,value,log,torch.as_tensor(reward,device=cfg['device']),torch.as_tensor(starts,device=cfg['device'],dtype=torch.float32)))
            if i%50==0:progress('rollout',completed=i,total=steps)
        torch.cuda.synchronize();rollout_s=time.perf_counter()-t_rollout
        t=time.perf_counter()
        with torch.no_grad():
            z,zv,_,_=rep.step_both(obs['image'],h,hv,starts)
            state=torch.as_tensor(obs['state'],device=cfg['device'])
            last=actor.distribution_value(torch.cat([z,state],-1),torch.cat([zv,state],-1))[1]
        features,critic_features,actions,values,old_log,rewards,dones=[torch.stack([row[i] for row in data]) for i in range(7)]
        advantages,returns=gae(rewards,values,dones,last,cfg['ppo']['gamma'],cfg['ppo']['gae_lambda'])
        torch.cuda.synchronize();gae_s=time.perf_counter()-t
        initial={k:v.detach().clone() for k,v in actor.state_dict().items()};updates=[]
        for batch in a.batch_sizes:
            progress('ppo_update',minibatch=batch)
            actor.load_state_dict(initial);optimizer=torch.optim.Adam(actor.parameters(),lr=cfg['ppo']['learning_rate'],eps=1e-5)
            update_cfg=copy.deepcopy(cfg['ppo']);update_cfg['batch_size']=batch
            torch.cuda.synchronize();t=time.perf_counter()
            metrics=ppo_update(actor,optimizer,features.flatten(0,1),critic_features.flatten(0,1),actions.flatten(0,1),old_log.flatten(),advantages.flatten(),returns.flatten(),update_cfg)
            torch.cuda.synchronize();elapsed=time.perf_counter()-t
            if not all(np.isfinite(x) for x in metrics.values()):raise FloatingPointError('Nonfinite benchmark update')
            updates.append({'minibatch':batch,'epochs':update_cfg['n_epochs'],'gradient_updates':int(np.ceil(steps*env.n/batch))*update_cfg['n_epochs'],
                'update_s':elapsed,'iteration_s':rollout_s+gae_s+elapsed,'effective_transitions_per_second':steps*env.n/(rollout_s+gae_s+elapsed),'metrics':metrics})
        result={'num_envs':env.n,'n_steps':steps,'samples':steps*env.n,'warmup_steps':a.warmup,'setup_warmup_s':setup_seconds,
                'rollout_s':rollout_s,'rollout_transitions_per_second':steps*env.n/rollout_s,'inference_s':inference_s,
                'environment_s':env_s,'gae_and_stack_s':gae_s,'completed_episodes':episodes,'timing':costs,'updates':updates,
                'torch_peak_mib':torch.cuda.max_memory_allocated()/2**20,
                'gpu_process_memory':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True).strip(),
                'scope':'Real stochastic-policy RTX rollout with actual reset/scene generation and 10-epoch source PPO updates. Approx. 64k samples per scale; frozen LSTM state persists across steps. Update variants restore the same actor; all benchmark weights discarded. Counts change task streams and temporal horizons, so this is performance evidence, not equal learning outcomes.'}
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');progress('completed')
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        progress('failed',error=repr(exc));raise
    finally:app.close()


if __name__=='__main__':main()
