import copy,json,unittest
from pathlib import Path
import numpy as np
from mavrl_isaac.dynamic_obstacles import MovingCylinders,move_existing_cylinders
from mavrl_isaac.environment import NavigationEnv
from mavrl_isaac.scenes import make_scene,segment_collision,feasible


def mover(a,b,speed=1.,phase=0.,radius=.2):
    return MovingCylinders(np.array([a],float),np.array([b],float),np.array([radius]),np.array([speed]),np.array([phase]))


class DynamicTests(unittest.TestCase):
    def test_crossing_between_frames(self):
        m=mover([-1,0],[1,0],20.)
        p=np.array([0.,0.,3.])
        self.assertFalse(segment_collision(p,p,m.at_time(0),.1))
        self.assertFalse(segment_collision(p,p,m.at_time(.1),.1))
        self.assertTrue(m.collision(p,p,0,.1,.1))

    def test_reversal_inside_action_step(self):
        m=mover([0,0],[1,0],20.)
        p=np.array([1.,0.,3.])
        np.testing.assert_allclose(m.at_time(0),m.at_time(.1))
        self.assertTrue(m.collision(p,p,0,.1,.1))

    def test_spatial_crossing_at_different_times_is_not_collision(self):
        m=mover([0,-2],[0,0],20.,radius=.05)
        self.assertFalse(m.collision(np.array([-1.,0,3]),np.array([1.,0,3]),0,.1,.05))

    def test_zero_speed_matches_static_collision(self):
        m=mover([-1,0],[1,0],0.,phase=1.)
        for y in [0.,.29,.31,1.]:
            a,b=np.array([-2.,y,3]),np.array([2.,y,3])
            self.assertEqual(m.collision(a,b,10.,10.1,.1),segment_collision(a,b,m.at_time(0),.1))

    def test_seeded_tracks_preserve_map_and_clear_endpoints(self):
        cfg=json.loads(Path('configs/isaac_capture2_dynamic.json').read_text())
        for density in ['sparse','medium','dense']:
            scene=make_scene(100200,cfg['scene'],.1,density)
            original=copy.deepcopy(scene)
            move_existing_cylinders(scene,cfg['scene'],cfg['dynamic_obstacles'],.1)
            repeat=move_existing_cylinders(copy.deepcopy(original),cfg['scene'],cfg['dynamic_obstacles'],.1)
            self.assertEqual(scene.moving.manifest(),repeat.moving.manifest())
            np.testing.assert_array_equal(scene.cylinders,original.cylinders)
            np.testing.assert_array_equal(scene.at_time(0),original.cylinders)
            self.assertEqual(len(scene.at_time(1.)),len(original.cylinders))
            np.testing.assert_array_equal(scene.moving.radii,original.cylinders[scene.moving.indices,2])
            np.testing.assert_array_equal(scene.start,original.start)
            np.testing.assert_array_equal(scene.goal,original.goal)
            self.assertTrue(feasible(scene.start,scene.goal,scene.at_time(0),40,40,.5,.1))
            for a,b,r,index in zip(scene.moving.starts,scene.moving.ends,scene.moving.radii,scene.moving.indices):
                self.assertFalse(segment_collision(a,b,np.delete(scene.cylinders,index,axis=0),r+.1))
                self.assertFalse(segment_collision(a,b,np.array([[*scene.start[:2],0],[*scene.goal[:2],0]]),r+1.5))

    def test_environment_uses_relative_motion_and_resets_time(self):
        cfg=json.loads(Path('configs/isaac_capture2.json').read_text());cfg['num_envs']=1
        class Renderer:
            def set_scene(self,i,scene):pass
            def update_dynamic(self,i,scene,t):self.last_time=t
            def render(self,p,r):return np.zeros((1,1,256,256),np.uint8)
        renderer=Renderer();env=NavigationEnv(cfg,renderer,density='empty');env.reset([123])
        env.position[0]=[0,10,3];env.scenes[0].moving=mover([-1,10],[1,10],20.)
        _,_,done,infos=env.step(np.zeros((1,4)),autoreset=False)
        self.assertTrue(done[0]);self.assertEqual(infos[0]['episode']['collision_type'],'dynamic')
        self.assertAlmostEqual(renderer.last_time,.1)
        env.reset([123]);self.assertEqual(env.elapsed[0],0.)

    def test_moved_obstacle_does_not_leave_static_collision_at_origin(self):
        cfg=json.loads(Path('configs/isaac_capture2.json').read_text());cfg['num_envs']=1
        env=NavigationEnv(cfg,density='empty');env.reset([123])
        env.position[0]=[0,10,3];env.elapsed[0]=1.
        env.scenes[0].cylinders=np.array([[0.,10.,.2]])
        env.scenes[0].moving=mover([-2,10],[2,10],1.,phase=2.)
        env.scenes[0].moving.indices=np.array([0])
        _,_,done,infos=env.step(np.zeros((1,4)),autoreset=False)
        self.assertFalse(done[0]);self.assertFalse(infos[0]['collision'])


if __name__=='__main__':unittest.main()
