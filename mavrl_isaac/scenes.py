"""Static cylinders, feasible missions, and source-style horizontal free distance."""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import label

@dataclass
class CylinderScene:
    cylinders: np.ndarray  # x, y, radius; z extent [0,height]
    start: np.ndarray
    goal: np.ndarray
    yaw: float
    density: str
    traversability: float
    seed: int
    moving: object = None

    def at_time(self,t):
        if self.moving is None:return self.cylinders
        if self.moving.indices is not None:
            if t==0:return self.cylinders
            cylinders=self.cylinders.copy()
            cylinders[self.moving.indices]=self.moving.at_time(t)
            return cylinders
        return np.concatenate((self.cylinders,self.moving.at_time(t)))

    @property
    def static_cylinders(self):
        if self.moving is None or self.moving.indices is None:return self.cylinders
        return np.delete(self.cylinders,self.moving.indices,axis=0)


def clearance(points,cylinders):
    if not len(cylinders):return np.full(np.shape(points)[:-1],np.inf)
    return np.min(np.linalg.norm(np.asarray(points)[...,None,:2]-cylinders[:,:2],axis=-1)-cylinders[:,2],axis=-1)


def segment_collision(start,end,cylinders,drone_radius):
    if not len(cylinders):return False
    d=end[:2]-start[:2]
    u=np.clip(((cylinders[:,:2]-start[:2])*d).sum(1)/max(d@d,1e-20),0,1)
    return bool(np.any(np.linalg.norm(start[:2]+u[:,None]*d-cylinders[:,:2],axis=1)<=cylinders[:,2]+drone_radius))


def feasible(start,goal,cylinders,width,length,grid,drone_radius):
    xs=np.arange(-width/2,width/2+grid/2,grid)
    ys=np.arange(0,length+grid/2,grid)
    points=np.stack(np.meshgrid(xs,ys,indexing='ij'),-1)
    # Conservative cell inflation: every point of each accepted cell is free.
    free=clearance(points,cylinders)>drone_radius+grid/2**0.5
    components,_=label(free)
    def index(p):return tuple(np.clip(np.round((p[:2]-[-width/2,0])/grid).astype(int),0,np.array(free.shape)-1))
    a,b=components[index(start)],components[index(goal)]
    return bool(a!=0 and a==b)


def traversability(cylinders,width,length):
    # Upstream averages 8 horizontal ray distances on a 1 m xyz grid.
    # Full-height cylinders make all z layers equal. Use exact circle hits in
    # place of 0.1 m point-cloud ray marches; preserve the measured quantity.
    x,y=np.meshgrid(np.arange(-width/2+.5,width/2,1.),np.arange(.5,length,1.))
    points=np.stack((x.ravel(),y.ravel()),axis=-1)
    free=clearance(points,cylinders)>0.1
    values=[]
    for angle in np.arange(8)*np.pi/4:
        d=np.array([np.sin(angle),np.cos(angle)])
        limits=[]
        for axis,(lo,hi) in enumerate([(-width/2,width/2),(0,length)]):
            if abs(d[axis])>1e-10:limits.append(((hi if d[axis]>0 else lo)-points[:,axis])/d[axis])
        hit=np.minimum.reduce(limits)
        if len(cylinders):
            q=points[:,None,:]-cylinders[None,:,:2]
            b=(q*d).sum(-1)
            c=(q*q).sum(-1)-(cylinders[:,2]+.1)**2
            discr=b*b-c
            near=-b-np.sqrt(np.maximum(discr,0))
            near=np.where((discr>=0)&(near>0),near,np.inf)
            hit=np.minimum(hit,near.min(1))
        values.append(np.where(free,hit,0.))
    result=float(np.mean(values))
    if result<=0:raise ValueError('Scene has zero traversability')
    return result


def make_scene(seed,cfg,drone_radius,density=None):
    rng=np.random.default_rng(seed)
    density=density or cfg['density']
    if density=='mixed':density=rng.choice(['sparse','medium','dense'])
    width,length=cfg['width'],cfg['length']
    for attempt in range(100):
        # Original forest scene uses three opposite-edge mission orientations.
        area=int(rng.integers(3));band=cfg['start_band'];goal_y=cfg['goal_y']
        if area==0:
            start=np.array([rng.uniform(-width/2+band,width/2-band),rng.uniform(0,band),3.])
            goal=np.array([rng.uniform(-width/2+band,width/2-band),goal_y-rng.uniform(0,band),3.+rng.random()]);yaw=0.
        elif area==1:
            start=np.array([-width/2+rng.uniform(0,band),rng.uniform(band,length),3.])
            goal=np.array([width/2-rng.uniform(0,band),rng.uniform(band,length-band),3.+rng.random()]);yaw=-np.pi/2
        else:
            start=np.array([width/2-rng.uniform(0,band),rng.uniform(0,length-band),3.])
            goal=np.array([-width/2+rng.uniform(0,band),rng.uniform(0,length-band),3.+rng.random()]);yaw=np.pi/2
        cylinders=[]
        if density!='empty':
            spacing=cfg['spacing'][density]
            target=int(rng.integers(max(1,int(.6*width*length/spacing**2)),max(2,int(.9*width*length/spacing**2))))
            centers=rng.uniform([-width/3,length/6],[width/3,5*length/6],size=(4,2))
            for _ in range(target*100):
                rr=cfg['large_radius_range'] if rng.random()<cfg['large_probability'] else cfg['radius_range']
                radius=rng.uniform(*rr)
                xy=rng.uniform([-width/2+radius,radius],[width/2-radius,length-radius])
                if cfg['distribution']=='clustered' and rng.random()<.7:
                    xy=centers[rng.integers(len(centers))]+rng.normal(0,spacing,2)
                    if not (-width/2+radius<xy[0]<width/2-radius and radius<xy[1]<length-radius):continue
                if min(np.linalg.norm(xy-start[:2]),np.linalg.norm(xy-goal[:2]))<radius+cfg['endpoint_clearance']:continue
                if cylinders:
                    c=np.asarray(cylinders)
                    required=radius+c[:,2]+2*drone_radius
                    if cfg['distribution']=='poisson':required=np.maximum(required,spacing)
                    if np.any(np.linalg.norm(c[:,:2]-xy,axis=1)<required):continue
                cylinders.append([*xy,radius])
                if len(cylinders)==target:break
        cylinders=np.asarray(cylinders,dtype=float).reshape(-1,3)
        if feasible(start,goal,cylinders,width,length,cfg['path_grid'],drone_radius):
            return CylinderScene(cylinders,start,goal,yaw,density,traversability(cylinders,width,length),seed)
    raise RuntimeError(f'No feasible mission generated for seed {seed}')
