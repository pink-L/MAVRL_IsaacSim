"""Plan all 50% subsets first, validate physical geometry, then freeze manifests."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from mavrl_isaac.scenes import make_scene,segment_collision
from mavrl_isaac.fraction_tracks import plan_master_tracks,fraction_count
from mavrl_isaac.dynamic_obstacles import MovingCylinders


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    if (root/'master_tracks.json').exists():raise ValueError('Use a new output directory')
    cfg=json.loads(Path('configs/isaac_capture2.json').read_text())
    motion={'enabled':True,'speed':1.,'track_length':4.,'seed_salt':20261106,'master_fraction':.5,
        'surface_gap':.001,'candidate_angles':1024,'planning_trials':32,
        'motion':'constant_speed_reflected_segment','selection':'original_cylinders',
        'radius_mode':'preserve_original','initial_phase':'track_midpoint','planner':'synchronous_exact_v1',
        'fraction_rounding':'floor(N*fraction+0.5)'}
    original=json.loads(Path('artifacts/dynamic_existing_20261106/baseline_full/scenes.json').read_text())
    tasks={};stats=[];started=time.perf_counter();minimum_gap=float('inf')
    for d_i,density in enumerate(['sparse','medium','dense']):
        for k in range(100):
            seed=100200+d_i*10000+k;key=f'{density}/{seed}'
            scene=make_scene(seed,cfg['scene'],.1,density)
            np.testing.assert_array_equal(scene.cylinders,np.array(original[key]['original_cylinders']))
            entry=plan_master_tracks(scene,cfg['scene'],motion);m=MovingCylinders(**{k:np.array(v) for k,v in entry['moving'].items()})
            np.testing.assert_allclose(m.at_time(0)[:,:2],scene.cylinders[m.indices,:2],atol=1e-12)
            np.testing.assert_array_equal(m.radii,scene.cylinders[m.indices,2])
            assert len(m.indices)==fraction_count(len(scene.cylinders),.5)
            for j,(a,b,r,index) in enumerate(zip(m.starts,m.ends,m.radii,m.indices)):
                assert not segment_collision(a,b,np.delete(scene.cylinders,index,axis=0),r+motion['surface_gap']*.99)
                assert not segment_collision(a,b,np.array([[*scene.start[:2],0],[*scene.goal[:2],0]]),r+1.5)
                # Independently check actual relative positions over the three
                # linear time intervals spanning the whole 8-second cycle.
                for t0,t1 in [(0.,2.),(2.,6.),(6.,8.)]:
                    q0=m.at_time(t0)[j,:2]-m.at_time(t0)[:j,:2]
                    q1=m.at_time(t1)[j,:2]-m.at_time(t1)[:j,:2]
                    delta=q1-q0;u=np.clip(-(q0*delta).sum(1)/np.maximum((delta*delta).sum(1),1e-20),0,1)
                    gap=np.linalg.norm(q0+u[:,None]*delta,axis=1)-r-m.radii[:j]
                    if len(gap):minimum_gap=min(minimum_gap,float(gap.min()));assert np.all(gap>motion['surface_gap']*.99)
            tasks[key]=entry
            stats.append({'density':density,'scene_seed':seed,'total':len(scene.cylinders),
                'dynamic_30':fraction_count(len(scene.cylinders),.3),'dynamic_50':len(m.indices),'planning_trial':entry['planning_trial']})
        print(density,'100 shared master tracks planned and independently validated',flush=True)
    manifest={'motion_config':motion,'tasks':tasks}
    path=root/'master_tracks.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    for fraction in [.3,.5]:
        config=json.loads(json.dumps(cfg));config['dynamic_obstacles']=motion|{'fraction':fraction,'manifest_path':str(path),'manifest_sha256':digest}
        Path(f'configs/isaac_capture2_dynamic_{int(fraction*100)}pct.json').write_text(json.dumps(config,indent=2)+'\n')
    result={'all_passed':True,'tasks':stats,'maps_checked':len(tasks),'master_manifest_sha256':digest,
        'minimum_moving_pair_surface_gap_m':minimum_gap,'elapsed_s':time.perf_counter()-started,
        'scope':'Original maps/radii/counts/start/goals preserved. Exact requested rounded counts; 30% is a fixed prefix of 50%. Full-cycle moving pairs checked; every track also avoids all other original stationary positions.'}
    (root/'geometry_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ALL_GEOMETRY_PASSED',json.dumps({k:v for k,v in result.items() if k!='tasks'}),flush=True)


if __name__=='__main__':main()
