import json,time,math
from pathlib import Path
import numpy as np
import torch
from .training import load_checkpoint,digest


def wilson(successes,n,z=1.96):
    p=successes/n;denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0.,center-half),min(1.,center+half)]


def evaluate(env,cfg,policy,out,episodes,densities,reset_memory=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rep,actor,checkpoint=load_checkpoint(policy,cfg['device']);actor.eval()
    # Test reward coefficients are recorded; success/collision are geometric.
    env.reward_phase='final';records=[];samples=[];summaries={};t0=time.perf_counter()
    dynamic=cfg.get('dynamic_obstacles',{}).get('enabled',False);scenes={}
    for density_index,density in enumerate(densities):
        env.density=density
        seeds=[cfg['seed']+100000+10000*density_index+i for i in range(episodes)]
        next_task=env.n;active=np.arange(env.n)<episodes
        obs=env.reset([seeds[i] if i<episodes else seeds[0] for i in range(env.n)])
        h=rep.zero_state(env.n);starts=np.ones(env.n,bool);results=[]
        while len(results)<episodes:
            with torch.no_grad():
                latent,h=rep.step(obs['image'],h,starts,reset_memory)
                action,_,_=actor.act(torch.cat([latent,torch.as_tensor(obs['state'],device=cfg['device'])],1),True)
            for i in np.flatnonzero(active):
                if env.steps[i]%10==0:
                    samples.append({'density':density,'scene_seed':env.scenes[i].seed,'time_s':float(env.elapsed[i]),
                        'speed':float(np.linalg.norm(env.velocity[i,:2])),'depth_gray_mean':float(obs['image'][i].mean()),
                        'position':env.position[i].tolist(),'action':action[i].cpu().tolist()})
                    if dynamic:
                        scene=env.scenes[i]
                        samples[-1]['dynamic_cylinders']=scene.moving.at_time(env.elapsed[i]).tolist()
                        samples[-1]['dynamic_velocities']=scene.moving.velocity_at(env.elapsed[i]).tolist()
                        if env.steps[i]==0:
                            scenes[f'{density}/{scene.seed}']={'density':density,'scene_seed':scene.seed,
                                'original_cylinders':scene.cylinders.tolist(),'static_cylinders':scene.static_cylinders.tolist(),'start':scene.start.tolist(),
                                'goal':scene.goal.tolist(),'yaw':scene.yaw,'moving':scene.moving.manifest()}
            obs,_,done,infos=env.step(action.cpu().numpy(),autoreset=False);starts=done
            needs_render=False
            for i in np.flatnonzero(done):
                if active[i]:
                    record=infos[i]['episode'];results.append(record);records.append(record)
                    print('EVAL',density,len(results),record['reason'],flush=True)
                    if next_task<episodes:env.reset_one(i,seeds[next_task]);next_task+=1
                    else:active[i]=False;env.reset_one(i,seeds[0])
                    needs_render=True
                else:
                    env.reset_one(i,seeds[0]);needs_render=True
            if needs_render:obs=env.observe()
        success=sum(r['reason']=='success' for r in results)
        summaries[density]={'episodes':len(results),'success_rate':success/len(results),
            'success_95ci':wilson(success,len(results)),
            'collision_rate':sum(r['reason']=='collision' for r in results)/len(results),
            'boundary_rate':sum(r['reason']=='boundary' for r in results)/len(results),
            'timeout_rate':sum(r['reason']=='timeout' for r in results)/len(results),
            'mean_speed_all_episodes':float(np.mean([r['mean_speed'] for r in results])),
            'mean_speed_successful_episodes':float(np.mean([r['mean_speed'] for r in results if r['reason']=='success'])) if success else None,
            'seeds':seeds}
        if dynamic:
            summaries[density]['collision_counts_by_type']={kind:sum(r['reason']=='collision' and r['collision_type']==kind for r in results) for kind in ['static','dynamic','both']}
    (out/'episodes.json').write_text(json.dumps(records,indent=2));(out/'trajectories.json').write_text(json.dumps(samples))
    report={'policy':str(policy),'sha256':digest(policy),'training_timesteps':checkpoint.get('timesteps'),
        'reset_memory':reset_memory,'config':cfg,'elapsed_s':time.perf_counter()-t0,'results':summaries,
        'scope':'Isaac Sim static cylinders; smoke/short-training results are not proof of convergence or original AvoidBench reproduction.'}
    if dynamic:
        (out/'scenes.json').write_text(json.dumps(scenes,indent=2))
        report['scope']='Zero-shot evaluation of static-trained MAVRL with selected original cylinders moving. Obstacle count, radii and initial map retained. Paired map/track seeds; no retraining. One motion setting and one training seed. Original static TRAV retained for reward reporting.'
    (out/'summary.json').write_text(json.dumps(report,indent=2));print('EVALUATION_SUMMARY',json.dumps(summaries),flush=True)
    return report
