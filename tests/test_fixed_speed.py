"""Fixed-speed control changes only its final-phase reward, never motion."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
from mavrl_isaac.environment import NavigationEnv


class FixedSpeedTests(unittest.TestCase):
    def test_reward_phase_gate_and_unchanged_dynamics(self):
        cfg=json.loads(Path('configs/isaac_mavrl.json').read_text());cfg['num_envs']=1
        fixed=copy.deepcopy(cfg);fixed['reward'].update(speed_mode='target',target_speed=2.)
        for phase in ['initial','final']:
            for speed in [0.,2.,3.,4.]:
                left=NavigationEnv(cfg,density='empty',reward_phase=phase)
                right=NavigationEnv(fixed,density='empty',reward_phase=phase)
                for env in [left,right]:
                    env.reset([123]);env.position[0]=[0.,10.,3.];env.goal[0]=[0.,35.,3.]
                    env.rotation[0]=np.eye(3);env.yaw[0]=0.;env.velocity[0]=[speed,0.,0.]
                a=left.step(np.zeros((1,4)),autoreset=False)
                b=right.step(np.zeros((1,4)),autoreset=False)
                np.testing.assert_array_equal(a[0]['state'],b[0]['state'])
                np.testing.assert_array_equal(left.position,right.position)
                np.testing.assert_array_equal(a[2],b[2])
                ta=np.array(a[3][0]['reward_terms']);tb=np.array(b[3][0]['reward_terms'])
                np.testing.assert_array_equal(ta[[0,2,3,4,5]],tb[[0,2,3,4,5]])
                if phase=='initial':np.testing.assert_array_equal(a[1],b[1])
                else:
                    self.assertAlmostEqual(tb[1],-.05*abs(speed-2.))
                    self.assertAlmostEqual(float(b[1][0]-a[1][0]),tb[1]-ta[1],places=6)

    def test_terminal_reward_overrides_target_speed_cost(self):
        cfg=json.loads(Path('configs/isaac_mavrl.json').read_text());cfg['num_envs']=1
        cfg['reward'].update(speed_mode='target',target_speed=2.)
        env=NavigationEnv(cfg,density='empty',reward_phase='final');env.reset([123]);env.position[0]=env.goal[0]
        _,reward,done,info=env.step(np.zeros((1,4)),autoreset=False)
        self.assertTrue(done[0]);self.assertEqual(info[0]['episode']['reason'],'success')
        self.assertAlmostEqual(float(reward[0]),20/info[0]['episode']['traversability'],places=6)


if __name__=='__main__':unittest.main()
