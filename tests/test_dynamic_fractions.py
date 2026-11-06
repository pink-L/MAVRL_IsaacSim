import copy,json,unittest
from pathlib import Path
import numpy as np
from mavrl_isaac.scenes import make_scene
from mavrl_isaac.dynamic_obstacles import move_existing_cylinders
from mavrl_isaac.fraction_tracks import fraction_count,compatible_directions


class FractionTests(unittest.TestCase):
    def test_round_half_up(self):
        self.assertEqual(fraction_count(23,.5),12)
        self.assertEqual(fraction_count(75,.3),23)
        self.assertEqual(fraction_count(84,.5),42)

    def test_parallel_tracks_can_have_overlapping_envelopes(self):
        valid=compatible_directions(np.array([0.,0.]),.4,np.array([[0.,1.]]),
            np.array([[2.,0.]]),np.array([.4]),np.array([[0.,1.]]),2.,.001)
        self.assertTrue(valid[0])

    def test_approaching_tracks_are_rejected(self):
        valid=compatible_directions(np.array([0.,0.]),.4,np.array([[1.,0.]]),
            np.array([[2.,0.]]),np.array([.4]),np.array([[-1.,0.]]),2.,.001)
        self.assertFalse(valid[0])

    def test_fraction_subsets_share_exact_motion_and_initial_map(self):
        configs=[json.loads(Path(f'configs/isaac_capture2_dynamic_{p}pct.json').read_text()) for p in [30,50]]
        for density,seed in [('sparse',100200),('medium',110200),('dense',120206),('dense',120296)]:
            original=make_scene(seed,configs[0]['scene'],.1,density)
            scenes=[move_existing_cylinders(copy.deepcopy(original),c['scene'],c['dynamic_obstacles'],.1) for c in configs]
            k=fraction_count(len(original.cylinders),.3)
            self.assertEqual(len(scenes[0].moving.indices),k)
            self.assertEqual(len(scenes[1].moving.indices),fraction_count(len(original.cylinders),.5))
            np.testing.assert_array_equal(scenes[0].moving.indices,scenes[1].moving.indices[:k])
            for t in [0.,.1,2.,6.,8.,99.9]:
                np.testing.assert_array_equal(scenes[0].moving.at_time(t),scenes[1].moving.at_time(t)[:k])
            for scene in scenes:
                np.testing.assert_array_equal(scene.at_time(0),original.cylinders)
                np.testing.assert_array_equal(scene.moving.radii,original.cylinders[scene.moving.indices,2])

    def test_manifest_rejects_changed_original_geometry(self):
        cfg=json.loads(Path('configs/isaac_capture2_dynamic_30pct.json').read_text())
        scene=make_scene(100200,cfg['scene'],.1,'sparse');scene.cylinders[0,0]+=.1
        with self.assertRaisesRegex(ValueError,'Original map differs'):
            move_existing_cylinders(scene,cfg['scene'],cfg['dynamic_obstacles'],.1)


if __name__=='__main__':unittest.main()
