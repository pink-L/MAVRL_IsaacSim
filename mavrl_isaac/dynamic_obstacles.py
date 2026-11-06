"""Seeded, nonreactive cylinder tracks, driven by per-episode simulation time."""
from dataclasses import dataclass
import numpy as np
from .scenes import segment_collision


@dataclass
class MovingCylinders:
    starts: np.ndarray
    ends: np.ndarray
    radii: np.ndarray
    speeds: np.ndarray
    phases: np.ndarray  # distance along a full out-and-back cycle at t=0
    indices: np.ndarray = None  # original cylinder indices, if replacing geometry

    @property
    def lengths(self):
        return np.linalg.norm(self.ends-self.starts, axis=1)

    def at_time(self, t):
        lengths = self.lengths
        distance = (self.phases+self.speeds*t) % (2*lengths)
        fraction = np.minimum(distance, 2*lengths-distance)/lengths
        xy = self.starts+(self.ends-self.starts)*fraction[:, None]
        return np.column_stack((xy, self.radii))

    def velocity_at(self, t):
        lengths = self.lengths
        distance = (self.phases+self.speeds*t) % (2*lengths)
        sign = np.where(distance < lengths, 1., -1.)
        return (self.ends-self.starts)/lengths[:, None]*(self.speeds*sign)[:, None]

    def collision(self, start, end, t0, t1, drone_radius):
        """Exact swept disks for linearly interpolated UAV steps and tracks.

        Split at every track reversal, including reversals inside one 0.1 s
        action interval. Checking only endpoint obstacles would miss crossings.
        """
        cuts = [t0, t1]
        for length, speed, phase in zip(self.lengths, self.speeds, self.phases):
            if speed == 0:
                continue
            first = int(np.floor((phase+speed*t0)/length))+1
            last = int(np.ceil((phase+speed*t1)/length))
            cuts.extend((k*length-phase)/speed for k in range(first, last))
        cuts = sorted(set(cuts))
        for a, b in zip(cuts[:-1], cuts[1:]):
            u0, u1 = (a-t0)/(t1-t0), (b-t0)/(t1-t0)
            p0 = np.asarray(start[:2])+(end[:2]-start[:2])*u0
            p1 = np.asarray(start[:2])+(end[:2]-start[:2])*u1
            q0 = p0-self.at_time(a)[:, :2]
            q1 = p1-self.at_time(b)[:, :2]
            delta = q1-q0
            fraction = np.clip(-(q0*delta).sum(1)/np.maximum((delta*delta).sum(1), 1e-20), 0, 1)
            if np.any(np.linalg.norm(q0+fraction[:, None]*delta, axis=1) <= self.radii+drone_radius):
                return True
        return False

    def manifest(self):
        result={name: getattr(self, name).tolist() for name in ('starts', 'ends', 'radii', 'speeds', 'phases')}
        result['indices']=None if self.indices is None else self.indices.tolist()
        return result


def move_existing_cylinders(scene, scene_cfg, cfg, drone_radius):
    """Move selected original cylinders; preserve radii, count and t=0 map.

    Entire tracks avoid static geometry and mission endpoints. Disjoint bounding
    disks keep movers from intersecting each other. Original initial grid
    feasibility is retained; this is not a time-dependent reachability guarantee.
    """
    if 'fraction' in cfg:
        from .fraction_tracks import attach_fraction_tracks
        return attach_fraction_tracks(scene,scene_cfg,cfg)
    count = int(cfg['count'])
    speed, length = (float(cfg[k]) for k in ('speed', 'track_length'))
    if count < 1 or count>len(scene.cylinders) or not np.isfinite([speed, length]).all() or length <= 0 or speed < 0:
        raise ValueError('Invalid moving-cylinder configuration')
    rng = np.random.default_rng(np.random.SeedSequence([scene.seed, int(cfg['seed_salt'])]))
    width, extent = scene_cfg['width'], scene_cfg['length']
    starts, ends, centers, radii, indices = [], [], [], [], []
    for index in rng.permutation(len(scene.cylinders)):
        center=scene.cylinders[index,:2];radius=float(scene.cylinders[index,2])
        if centers and np.any(np.linalg.norm(np.array(centers)-center,axis=1)<=length+radius+np.array(radii)+2*drone_radius):
            continue
        others=np.delete(scene.cylinders,index,axis=0)
        for _ in range(128):
            angle=rng.uniform(0,2*np.pi)
            offset=length/2*np.array([np.cos(angle),np.sin(angle)])
            a,b=center-offset,center+offset
            if np.any(np.minimum(a,b)<[-width/2+radius,radius]) or np.any(np.maximum(a,b)>[width/2-radius,extent-radius]):
                continue
            if segment_collision(a,b,others,radius+drone_radius):
                continue
            endpoints=np.array([[*scene.start[:2],0.],[*scene.goal[:2],0.]])
            if segment_collision(a,b,endpoints,radius+scene_cfg['endpoint_clearance']):
                continue
            starts.append(a);ends.append(b);centers.append(center);radii.append(radius);indices.append(int(index))
            break
        if len(starts)==count:
            scene.moving=MovingCylinders(np.array(starts),np.array(ends),np.array(radii),
                np.full(count,speed),np.full(count,length/2),np.array(indices))
            return scene
    raise RuntimeError(f'Could not select {count} movable cylinders for {scene.density}/{scene.seed}; no task was silently dropped')
