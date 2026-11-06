"""Shared nested moving subsets, with exact synchronous track separation."""
from functools import lru_cache
import hashlib,json
from pathlib import Path
import numpy as np


def fraction_count(total,fraction):
    if total<1 or not 0<fraction<=1:raise ValueError('Invalid dynamic fraction')
    return max(1,int(np.floor(total*fraction+.5)))


def compatible_directions(center,radius,directions,other_centers,other_radii,other_directions,half_length,gap):
    """All tracks have the same length/speed/phase: relative position is q+u*d.

    A complete out-and-back cycle spans every u in [-half_length,half_length].
    Spatially overlapping track envelopes are fine if actual disks never touch.
    """
    if not len(other_centers):return np.ones(len(directions),bool)
    q=center-np.asarray(other_centers)
    d=directions[:,None,:]-np.asarray(other_directions)[None,:,:]
    u=np.clip(-(q[None]*d).sum(-1)/np.maximum((d*d).sum(-1),1e-20),-half_length,half_length)
    distance=np.linalg.norm(q[None]+u[:,:,None]*d,axis=-1)
    return np.all(distance>radius+np.asarray(other_radii)[None,:]+gap,axis=1)


def _directions(scene,scene_cfg,index,cfg):
    cylinders=scene.cylinders;center=cylinders[index,:2];radius=cylinders[index,2]
    length=cfg['track_length'];half=length/2;gap=cfg['surface_gap']
    rng=np.random.default_rng(np.random.SeedSequence([scene.seed,cfg['seed_salt'],int(index),1]))
    angles=rng.uniform(0,2*np.pi,cfg['candidate_angles'])
    directions=np.column_stack((np.cos(angles),np.sin(angles)))
    a=center-half*directions;b=center+half*directions
    lo=np.array([-scene_cfg['width']/2+radius,radius]);hi=np.array([scene_cfg['width']/2-radius,scene_cfg['length']-radius])
    valid=np.all(np.minimum(a,b)>=lo,axis=1)&np.all(np.maximum(a,b)<=hi,axis=1)
    # Avoid all other ORIGINAL positions as well as actual moving/moving
    # overlaps. Any prefix can consequently leave the remaining cylinders still.
    others=np.delete(cylinders,index,axis=0)
    points=np.concatenate((others[:,:2],np.array([scene.start[:2],scene.goal[:2]])))
    limits=np.concatenate((radius+others[:,2]+gap,np.full(2,radius+scene_cfg['endpoint_clearance'])))
    delta=b-a;q=points[None,:,:]-a[:,None,:]
    u=np.clip((q*delta[:,None,:]).sum(-1)/(length*length),0,1)
    distance=np.linalg.norm(a[:,None,:]+u[:,:,None]*delta[:,None,:]-points[None,:,:],axis=-1)
    valid &= np.all(distance>limits[None,:],axis=1)
    return directions[valid]


def plan_master_tracks(scene,scene_cfg,cfg):
    count=fraction_count(len(scene.cylinders),cfg['master_fraction'])
    if cfg['speed']<=0 or cfg['track_length']<=0:raise ValueError('Motion must have positive length and speed')
    candidates=[_directions(scene,scene_cfg,i,cfg) for i in range(len(scene.cylinders))]
    half=cfg['track_length']/2;best=0
    for trial in range(cfg['planning_trials']):
        rng=np.random.default_rng(np.random.SeedSequence([scene.seed,cfg['seed_salt'],trial,2]))
        indices=[];directions=[]
        for index in rng.permutation(len(scene.cylinders)):
            options=candidates[index]
            if not len(options):continue
            options=options[rng.permutation(len(options))]
            center=scene.cylinders[index,:2];radius=scene.cylinders[index,2]
            valid=compatible_directions(center,radius,options,scene.cylinders[indices,:2],
                scene.cylinders[indices,2],directions,half,cfg['surface_gap'])
            if not valid.any():continue
            indices.append(int(index));directions.append(options[np.flatnonzero(valid)[0]])
            if len(indices)==count:
                centers=scene.cylinders[indices,:2];directions=np.array(directions)
                return {'density':scene.density,'scene_seed':scene.seed,'original_cylinders':scene.cylinders.tolist(),
                    'start':scene.start.tolist(),'goal':scene.goal.tolist(),'planning_trial':trial,
                    'moving':{'starts':(centers-half*directions).tolist(),'ends':(centers+half*directions).tolist(),
                        'radii':scene.cylinders[indices,2].tolist(),'speeds':[cfg['speed']]*count,
                        'phases':[half]*count,'indices':indices}}
        best=max(best,len(indices))
    raise RuntimeError(f'Insufficient nonintersecting tracks for {scene.density}/{scene.seed}: {best}/{count}; task retained as failure')


@lru_cache(maxsize=4)
def _manifest(path,expected_sha):
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected_sha:raise ValueError('Motion manifest SHA256 mismatch')
    return json.loads(data)


def attach_fraction_tracks(scene,scene_cfg,cfg):
    from .dynamic_obstacles import MovingCylinders
    count=fraction_count(len(scene.cylinders),cfg['fraction'])
    if cfg['fraction']>cfg['master_fraction']:raise ValueError('Fraction exceeds the shared master subset')
    if cfg.get('manifest_path'):
        manifest=_manifest(cfg['manifest_path'],cfg['manifest_sha256'])
        for key in ['speed','track_length','seed_salt','master_fraction','surface_gap','candidate_angles','planning_trials']:
            if cfg[key]!=manifest['motion_config'][key]:raise ValueError('Motion manifest configuration mismatch: '+key)
        entry=manifest['tasks'][f'{scene.density}/{scene.seed}']
    else:entry=plan_master_tracks(scene,scene_cfg,cfg)
    if not np.array_equal(scene.cylinders,np.asarray(entry['original_cylinders'])):
        raise ValueError('Original map differs from the planned motion manifest')
    if not np.array_equal(scene.start,entry['start']) or not np.array_equal(scene.goal,entry['goal']):
        raise ValueError('Mission differs from the planned motion manifest')
    motion=entry['moving']
    if count>len(motion['indices']):raise ValueError('Insufficient motion tracks; refusing to reduce the requested count')
    scene.moving=MovingCylinders(**{key:np.asarray(values[:count]) for key,values in motion.items()})
    return scene
