import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .dynamics import integrate,observation,progress_reward
from .scenes import make_scene,segment_collision

class NavigationEnv:
    def __init__(self,cfg,renderer=None,density=None,reward_phase='initial'):
        self.cfg=copy.deepcopy(cfg);self.n=cfg['num_envs'];self.renderer=renderer
        self.rng=np.random.default_rng(cfg['seed']);self.density=density;self.reward_phase=reward_phase
        self.position=np.zeros((self.n,3));self.velocity=np.zeros_like(self.position)
        self.accel=np.zeros_like(self.position);self.rotation=np.tile(np.eye(3),(self.n,1,1))
        self.goal=np.zeros_like(self.position);self.yaw=np.zeros(self.n);self.elapsed=np.zeros(self.n)
        self.episode_return=np.zeros(self.n);self.speed_sum=np.zeros(self.n);self.distance=np.zeros(self.n)
        self.steps=np.zeros(self.n,dtype=int);self.scenes=[None]*self.n
        self.last_image=None;self.episode_records=[];self.current_starts=np.ones(self.n,dtype=bool)

    def reset_one(self,i,seed=None):
        seed=int(self.rng.integers(0,2**31-1)) if seed is None else int(seed)
        scene=make_scene(seed,self.cfg['scene'],self.cfg['simulation']['drone_radius'],self.density)
        if self.cfg.get('dynamic_obstacles',{}).get('enabled',False):
            from .dynamic_obstacles import move_existing_cylinders
            move_existing_cylinders(scene,self.cfg['scene'],self.cfg['dynamic_obstacles'],self.cfg['simulation']['drone_radius'])
        self.scenes[i]=scene;self.position[i]=scene.start;self.goal[i]=scene.goal
        self.velocity[i]=0;self.accel[i]=0;self.yaw[i]=scene.yaw
        self.rotation[i]=Rotation.from_euler('z',scene.yaw).as_matrix()
        self.elapsed[i]=0;self.episode_return[i]=0;self.speed_sum[i]=0;self.distance[i]=0;self.steps[i]=0
        if self.renderer:self.renderer.set_scene(i,scene)

    def observe(self):
        state=observation(self.position,self.velocity,self.goal,self.yaw).astype(np.float32)
        if self.renderer:
            for i,scene in enumerate(self.scenes):
                if scene.moving is not None:self.renderer.update_dynamic(i,scene,self.elapsed[i])
            self.last_image=self.renderer.render(self.position,self.rotation)
        # None backend exists only for numerical unit tests, never for experiment results.
        else:self.last_image=np.full((self.n,1,256,256),255,dtype=np.uint8)
        return {'image':self.last_image,'state':state}

    def reset(self,seeds=None):
        for i in range(self.n):self.reset_one(i,None if seeds is None else seeds[i])
        self.current_starts[:]=True
        return self.observe()

    def step(self,actions,autoreset=True):
        sim,rew=self.cfg['simulation'],self.cfg['reward']
        actions=np.asarray(actions,dtype=np.float64)
        if actions.shape!=(self.n,4) or not np.all(np.isfinite(actions)):raise ValueError('Invalid action batch')
        physical=np.clip(actions,-1,1)*sim['action_scale']
        old_pos=self.position.copy();old_accel=self.accel.copy()
        self.position,self.velocity,self.rotation,self.yaw=integrate(self.position,self.velocity,self.rotation,physical,sim['dt'])
        self.accel=physical[:,:3].copy();self.elapsed+=sim['dt'];self.steps+=1
        obs=observation(self.position,self.velocity,self.goal,self.yaw)
        rewards,terms=progress_reward(obs,self.accel,old_accel,rew[self.reward_phase],rew['speed_threshold'])
        # Explicit paper-style fixed-speed ablation. The baseline path above
        # stays unchanged; only the final-phase speed term is replaced.
        if self.reward_phase=='final' and rew.get('speed_mode','threshold')=='target':
            target=float(rew['target_speed'])
            if not np.isfinite(target) or target<=0:raise ValueError('target_speed must be finite and positive')
            old_speed_term=terms[:,1].copy()
            terms[:,1]=rew['final'][1]*abs(obs[:,1]-target)
            rewards+=terms[:,1]-old_speed_term
        dones=np.zeros(self.n,dtype=bool);infos=[]
        self.speed_sum+=np.linalg.norm(self.velocity[:,:2],axis=1)
        self.distance+=np.linalg.norm(self.position-old_pos,axis=1)
        bounds=np.asarray(sim['bounds'])
        for i in range(self.n):
            scene=self.scenes[i]
            static_collision=segment_collision(old_pos[i],self.position[i],scene.static_cylinders,sim['drone_radius'])
            dynamic_collision=scene.moving is not None and scene.moving.collision(
                old_pos[i],self.position[i],self.elapsed[i]-sim['dt'],self.elapsed[i],sim['drone_radius'])
            collision=static_collision or dynamic_collision
            reason=None
            if np.any(self.position[i]<bounds[:,0]) or np.any(self.position[i]>bounds[:,1]):reason='boundary';rewards[i]=rew['boundary']
            elif self.elapsed[i]>sim['max_time']+1e-8:reason='timeout';rewards[i]=rew['timeout']
            elif collision and sim['reset_if_collide']:reason='collision';rewards[i]=rew['collision']
            elif np.all(abs(self.position[i]-self.goal[i])<rew['goal_tolerance']):reason='success';rewards[i]=rew['arrival_scale']/self.scenes[i].traversability
            self.episode_return[i]+=rewards[i]
            info={'reward_terms':terms[i].tolist(),'collision':collision}
            if reason:
                dones[i]=True
                record={'scene_seed':self.scenes[i].seed,'density':self.scenes[i].density,'reason':reason,
                    'return':float(self.episode_return[i]),'steps':int(self.steps[i]),'duration_s':float(self.elapsed[i]),
                    'mean_speed':float(self.speed_sum[i]/self.steps[i]),'path_length':float(self.distance[i]),
                    'traversability':self.scenes[i].traversability,'num_obstacles':len(self.scenes[i].cylinders)}
                if scene.moving is not None:
                    record.update(num_obstacles=len(scene.at_time(self.elapsed[i])),
                        num_static_obstacles=len(scene.static_cylinders),num_dynamic_obstacles=len(scene.moving.radii),
                        collision_type=('both' if static_collision and dynamic_collision else 'dynamic' if dynamic_collision else 'static') if collision else None)
                info['episode']=record;self.episode_records.append(record)
                if autoreset:self.reset_one(i)
            infos.append(info)
        self.current_starts=dones.copy()
        return self.observe(),rewards.astype(np.float32),dones,infos
