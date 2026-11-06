"""Isaac Sim RTX depth cameras. Import only after SimulationApp startup."""
import numpy as np
from scipy.spatial.transform import Rotation
from .dynamics import FLU_TO_RFU,preprocess_depth


def launch(headless=True,runtime_log=None):
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    runtime_log=Path(runtime_log).resolve() if runtime_log is not None else root/'logs/isaac_runtime.log'
    if '--portable-root' not in sys.argv:sys.argv+=['--portable-root',str(root/'.runtime/kit')]
    from isaacsim import SimulationApp
    return SimulationApp({'headless':headless,'multi_gpu':False,'width':256,'height':256,
        'anti_aliasing':0,'limit_cpu_threads':4,'enable_crashreporter':False,
        'extra_args':['--/log/level=error','--/log/file='+str(runtime_log),
        '--/rtx/post/dlss/execMode=2','--/rtx/post/aa/op=0']})


class IsaacDepthRenderer:
    def __init__(self,cfg):
        from isaacsim.core.api import World
        from isaacsim.sensors.camera import Camera,CameraView
        from pxr import UsdGeom,Gf,UsdLux
        import omni.usd
        import omni.replicator.core as rep
        self.rep=rep;self.UsdGeom=UsdGeom;self.Gf=Gf
        self.cfg=cfg;self.camera_cfg=cfg['camera'];self.n=cfg['num_envs']
        self.world=World(physics_dt=.1,rendering_dt=.1,stage_units_in_meters=1.)
        self.stage=omni.usd.get_context().get_stage()
        self.origins=np.array([[100.*i,0.,0.] for i in range(self.n)])
        self.cameras=[];self.objects=[[] for _ in range(self.n)]
        UsdLux.DomeLight.Define(self.stage,'/World/light').CreateIntensityAttr(500.)
        for i in range(self.n):
            base=f'/World/env_{i}'
            floor=UsdGeom.Cube.Define(self.stage,base+'/floor');floor.CreateSizeAttr(1.)
            x=UsdGeom.Xformable(floor);x.AddTranslateOp().Set(Gf.Vec3d(*(self.origins[i]+[0,20,-.1])))
            x.AddScaleOp().Set(Gf.Vec3f(44,48,.2))
            cam=Camera(base+'/camera',resolution=(self.camera_cfg['width'],self.camera_cfg['height']))
            cam.set_focal_length(1.)
            cam.set_horizontal_aperture(2*np.tan(np.deg2rad(self.camera_cfg['fov']/2)))
            cam.set_clipping_range(.01,self.camera_cfg['max_depth']+.1)
            self.cameras.append(cam)
        self.world.reset()
        self.tiled=cfg['camera'].get('tiled',False)
        if self.tiled:
            self.view=CameraView(prim_paths_expr=[c.prim_path for c in self.cameras],camera_resolution=(256,256),output_annotators=['depth'],reset_xform_properties=False)
            self.order=[self.view.prim_paths.index(c.prim_path) for c in self.cameras]
        else:
            for cam in self.cameras:
                cam.initialize(attach_rgb_annotator=False)
                cam.add_distance_to_image_plane_to_frame()

    def set_scene(self,i,scene):
        UsdGeom,Gf=self.UsdGeom,self.Gf
        objects=self.objects[i]
        cylinders=scene.at_time(0.)
        while len(objects)<len(cylinders):
            cyl=UsdGeom.Cylinder.Define(self.stage,f'/World/env_{i}/cylinder_{len(objects)}')
            cyl.CreateAxisAttr('Z');cyl.CreateHeightAttr(self.cfg['scene']['height'])
            op=UsdGeom.Xformable(cyl).AddTranslateOp()
            objects.append((cyl,op))
        for j,(cyl,op) in enumerate(objects):
            if j<len(cylinders):
                x,y,r=cylinders[j];cyl.CreateRadiusAttr(float(r))
                op.Set(Gf.Vec3d(*(self.origins[i]+[x,y,self.cfg['scene']['height']/2])))
                cyl.GetVisibilityAttr().Set('inherited')
            else:cyl.GetVisibilityAttr().Set('invisible')

    def update_dynamic(self,i,scene,t):
        indices=scene.moving.indices
        if indices is None:indices=range(len(scene.cylinders),len(scene.cylinders)+len(scene.moving.radii))
        cylinders=scene.at_time(t)
        for j in indices:
            x,y,_=cylinders[j]
            self.objects[i][j][1].Set(self.Gf.Vec3d(*(self.origins[i]+[x,y,self.cfg['scene']['height']/2])))

    def render(self,positions,rotations):
        for i,cam in enumerate(self.cameras):
            rot=FLU_TO_RFU@rotations[i]
            pos=positions[i]+self.origins[i]+rot@np.asarray(self.camera_cfg['offset_flu'])
            xyzw=Rotation.from_matrix(rot).as_quat()
            cam.set_world_pose(pos,xyzw[[3,0,1,2]],camera_axes='world')
        # Explicit synchronous capture: wait for RTX and annotators before read.
        # Extra render subframes hold the simulation state fixed. Their count
        # belongs to the sensor adapter, not the 0.1 s kinematic action step.
        self.rep.orchestrator.step(rt_subframes=int(self.camera_cfg.get('rt_subframes',1)),
                                   delta_time=.1,pause_timeline=False,wait_for_render=True)
        if self.tiled:
            depth=self.view.get_depth()
            if tuple(depth.shape)!=(self.n,256,256,1):raise RuntimeError(f'Invalid tiled depth {depth.shape}')
            # CameraView reshapes the annotator buffer with a Warp CUDA kernel,
            # then exposes it to Torch without copying. Complete producer work
            # before Torch copies it to CPU or the next render reuses buffers.
            if self.camera_cfg.get('synchronize_depth',True):
                import warp as wp
                wp.synchronize_device(str(depth.device))
            return preprocess_depth(depth.cpu().numpy()[self.order,:,:,0],self.camera_cfg['max_depth'])[:,None]
        data=[]
        for camera in self.cameras:
            depth=camera.get_depth()
            if depth is None or depth.shape!=(256,256):
                raise RuntimeError(f'Invalid Isaac depth frame: {None if depth is None else depth.shape}')
            data.append(preprocess_depth(depth,self.camera_cfg['max_depth']))
        return np.stack(data)[:,None]
