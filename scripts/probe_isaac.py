from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.argv += ['--portable-root',str(ROOT/'.runtime/kit')]
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'multi_gpu':False,'width':256,'height':256,'anti_aliasing':0,'limit_cpu_threads':4,'enable_crashreporter':False,'extra_args':['--/log/file='+str(ROOT/'logs/isaac_kit.log')]})
import numpy as np
from pxr import UsdGeom,Gf
from isaacsim.core.api import World
from isaacsim.sensors.camera import Camera
world=World(physics_dt=0.1,rendering_dt=0.1,stage_units_in_meters=1.0)
import omni.usd
stage=omni.usd.get_context().get_stage()
cyl=UsdGeom.Cylinder.Define(stage,'/World/cylinder')
cyl.CreateRadiusAttr(1.0);cyl.CreateHeightAttr(6.0);cyl.CreateAxisAttr('Z')
UsdGeom.Xformable(cyl).AddTranslateOp().Set(Gf.Vec3d(5,0,3))
cam=Camera('/World/camera',position=np.array([0.,0.,3.]),resolution=(256,256))
cam.set_world_pose(np.array([0.,0.,3.]),np.array([1.,0.,0.,0.]),camera_axes='world')
cam.set_focal_length(10.0);cam.set_horizontal_aperture(20.0);cam.set_clipping_range(0.1,100.0)
world.reset();cam.initialize(attach_rgb_annotator=False);cam.add_distance_to_image_plane_to_frame()
import omni.replicator.core as rep
for _ in range(4):
 world.step(render=False)
 rep.orchestrator.step(rt_subframes=1,delta_time=0.1,pause_timeline=False)
depth=cam.get_depth()
assert depth is not None and depth.shape==(256,256),str(None if depth is None else depth.shape)
np.save(ROOT/'artifacts/probe_depth.npy',depth)
center=float(depth[128,128]);print('ISAAC_PROBE',depth.shape,center,np.isfinite(depth).sum(),flush=True)
assert abs(center-4)<0.08,center
cam.set_world_pose(np.array([1.,0.,3.]),np.array([1.,0.,0.,0.]),camera_axes='world')
rep.orchestrator.step(rt_subframes=1,delta_time=0.1,pause_timeline=False)
print('ISAAC_PROBE_MOVED',float(cam.get_depth()[128,128]),flush=True)
app.close()
