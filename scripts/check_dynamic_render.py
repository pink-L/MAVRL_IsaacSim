"""Actual RTX probe: moving geometry must change depth and replay at fixed time."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path('configs/isaac_capture2.json').read_text());cfg['num_envs']=1
    from mavrl_isaac.renderer import launch,IsaacDepthRenderer
    app=launch(runtime_log=out/'isaac_runtime.log')
    try:
        from mavrl_isaac.scenes import CylinderScene
        from mavrl_isaac.dynamic_obstacles import MovingCylinders
        renderer=IsaacDepthRenderer(cfg)
        scene=CylinderScene(np.array([[0.,6.,.5]]),np.array([0.,0.,3.]),np.array([0.,30.,3.]),0.,'probe',10.,0)
        scene.moving=MovingCylinders(np.array([[-2.,6.]]),np.array([[2.,6.]]),np.array([.5]),np.array([1.]),np.array([2.]),np.array([0]))
        renderer.set_scene(0,scene)
        frames=[];times=[0.,1.,2.,0.]
        for t in times:
            renderer.update_dynamic(0,scene,t)
            frames.append(renderer.render(scene.start[None],np.eye(3)[None])[0,0])
        np.savez_compressed(out/'frames.npz',frames=np.array(frames),times=times)
        center=[int(frame[128,128]) for frame in frames]
        result={'times':times,'center_gray':center,'center_depth_m':[v*12/255 for v in center],
            'fixed_time_replay_exact':bool(np.array_equal(frames[0],frames[3])),
            'motion_changes_depth':bool(np.any(frames[0]!=frames[2])),
            'geometry_count':len(renderer.objects[0]),
            'center_obstacle_then_clear':bool(4.8<center[0]*12/255<5.8 and center[2]==255)}
        (out/'result.json').write_text(json.dumps(result,indent=2))
        print('DYNAMIC_RENDER_CHECK',json.dumps(result),flush=True)
        if not all(result[k] for k in ['fixed_time_replay_exact','motion_changes_depth','center_obstacle_then_clear']):
            raise RuntimeError('Dynamic RTX probe failed')
    finally:app.close()


if __name__=='__main__':main()
