"""Reproducible independent-camera / tiled-camera pose and reset comparison."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--tiled',action='store_true')
 p.add_argument('--compare',nargs=2);p.add_argument('--allow-boundary-differences',action='store_true')
 a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 if a.compare:
  left,right=[Path(x) for x in a.compare];x=np.load(left/'frames.npz');y=np.load(right/'frames.npz')
  from scipy.ndimage import minimum_filter,maximum_filter
  rows=[]
  for name in x.files:
   delta=np.abs(x[name].astype(float)-y[name].astype(float))
   # Characterize, do not conceal, one-pixel depth-discontinuity differences.
   # The large-error pixel must fall inside the other image's 3x3 depth range
   # in BOTH directions; all other pixels retain the original 2-gray tolerance.
   lo_x=minimum_filter(x[name],size=(1,1,3,3));hi_x=maximum_filter(x[name],size=(1,1,3,3))
   lo_y=minimum_filter(y[name],size=(1,1,3,3));hi_y=maximum_filter(y[name],size=(1,1,3,3))
   boundary=(y[name]>=lo_x)&(y[name]<=hi_x)&(x[name]>=lo_y)&(x[name]<=hi_y)&(delta>2)
   rows.append({'case':name,'mae_gray':float(delta.mean()),'max_abs_gray':float(delta.max()),'equal_fraction':float((delta==0).mean()),
    'boundary_large_error_pixels':int(boundary.sum()),'boundary_large_error_fraction':float(boundary.mean()),
    'nonboundary_max_abs_gray':float(delta[~boundary].max())})
  strict=all(row['max_abs_gray']<=2 and row['mae_gray']<.2 for row in rows)
  approximate=all(row['nonboundary_max_abs_gray']<=2 and row['boundary_large_error_fraction']<=1e-4 and row['mae_gray']<.2 for row in rows)
  result={'captures':[{ 'path':str(d),'frames_sha256':sha(d/'frames.npz'),'metadata_sha256':sha(d/'metadata.json')} for d in [left,right]],'comparisons':rows,
   'strict_pixel_test_pass':strict,'boundary_characterized_test_pass':approximate,'allow_boundary_differences':a.allow_boundary_differences,
   'comparison_script_sha256':sha(__file__),
   'scope':'Geometric pose checks run during each capture. Relaxed comparison explicitly permits <=0.01% depth-boundary pixels within mutual 3x3 depth ranges. It is not pixel-perfect equivalence. Precise renderer numerical cause is unverified.'}
  (out/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
  assert approximate if a.allow_boundary_differences else strict
  return
 from scripts.run import record_run
 cfg=json.loads(Path('configs/isaac_mavrl.json').read_text());cfg['num_envs']=16;cfg['camera']['tiled']=a.tiled
 record_run(out,cfg,a)
 from mavrl_isaac.renderer import launch,IsaacDepthRenderer
 from mavrl_isaac.scenes import CylinderScene
 from mavrl_isaac.dynamics import FLU_TO_RFU
 app=launch();frames={};checks=[]
 try:
  renderer=IsaacDepthRenderer(cfg);n=cfg['num_envs'];positions=np.tile([0.,0.,3.],(n,1));rotations=np.tile(np.eye(3),(n,1,1))
  cylinders=[np.array([[0.,5.+.2*i,.6+.02*i]]) for i in range(n)]
  def scene(i,c):return CylinderScene(c,positions[i],np.array([0.,30.,3.]),0.,'fixture',1.,i)
  for i,c in enumerate(cylinders):renderer.set_scene(i,scene(i,c))
  def capture(name):frames[name]=renderer.render(positions,rotations)
  capture('base')
  for i in range(n):
   # Optical center is .1m laterally displaced by the configured left camera.
   expected=cylinders[i][0,1]-.2-np.sqrt(cylinders[i][0,2]**2-.1**2)
   observed=float(frames['base'][i,0,127:129,127:129].mean()/255*12)
   checks.append({'environment':i,'expected_center_depth_m':float(expected),'observed_quantized_depth_m':observed})
   assert abs(expected-observed)<.08,(i,expected,observed)
  positions[:,1]+=1;capture('forward_1m')
  assert np.all(frames['forward_1m'][:,0,128,128]<frames['base'][:,0,128,128]-15)
  rotations[:]=Rotation.from_euler('z',np.pi/2).as_matrix();capture('yaw_90deg')
  assert np.all(frames['yaw_90deg'][:,0,128,128]==255)
  rotations[:]=Rotation.from_euler('xyz',[.08,-.12,.17]).as_matrix();capture('pitch_roll_yaw')
  rotations[:]=np.eye(3)
  for i,c in enumerate(cylinders):
   updated=np.empty((0,3)) if i%2==0 else c+np.array([[0.,1.,0.]])
   renderer.set_scene(i,scene(i,updated))
  capture('reset_geometry')
  assert np.all(frames['reset_geometry'][::2,0,128,128]==255)
  assert np.all(frames['reset_geometry'][1::2,0,128,128]<255)
  np.savez_compressed(out/'frames.npz',**frames)
  metadata={'tiled':a.tiled,'config':cfg,'analytic_center_checks':checks,
   'scope':'Isaac tiled/individual renderer equivalence, camera pose updates, environment order and geometry reset. Does not prove Flightmare stereo/full-pose equivalence.',
   'frames_sha256':sha(out/'frames.npz')}
  (out/'metadata.json').write_text(json.dumps(metadata,indent=2));print('CAMERA_CHECKS_PASS',a.tiled)
 finally:app.close()

if __name__=='__main__':main()
