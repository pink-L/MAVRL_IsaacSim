"""NumPy port of AvoidVisionEnv::step/getObs, including its frame convention.

Position is RFU; stored velocity/acceleration are FLU. The original keeps body
velocity components unchanged on yaw updates (not rigid-body inertial dynamics).
"""
import numpy as np
from scipy.spatial.transform import Rotation
FLU_TO_RFU = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])

def wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

def attitude(accel, yaw):
    thrust = np.stack((-accel[:,1], accel[:,0], accel[:,2]+9.81),axis=1)
    thrust /= np.linalg.norm(thrust,axis=1,keepdims=True)
    # Eigen::Quaternion::FromTwoVectors(UnitZ, thrust), xyzw order.
    q = np.stack((-thrust[:,1],thrust[:,0],np.zeros(len(yaw)),1+thrust[:,2]),axis=1)
    q /= np.linalg.norm(q,axis=1,keepdims=True)
    return (Rotation.from_quat(q)*Rotation.from_euler('z',yaw)).as_matrix()

def integrate(position, velocity, rotation, action, dt):
    accel, yaw_rate=action[:,:3],action[:,3]
    world_rotation=np.einsum('ij,njk->nik',FLU_TO_RFU,rotation)
    new_position=position+np.einsum('nij,nj->ni',world_rotation,velocity*dt+0.5*accel*dt*dt)
    new_velocity=velocity+accel*dt
    # Original quaternionToEuler computes yaw from the previous full attitude.
    yaw=wrap(np.arctan2(rotation[:,1,0],rotation[:,0,0])+yaw_rate*dt)
    return new_position,new_velocity,attitude(accel,yaw),yaw

def observation(position,velocity,goal,yaw):
    delta=goal-position
    return np.stack((np.log1p(np.linalg.norm(delta[:,:2],axis=1)),
        np.linalg.norm(velocity[:,:2],axis=1),np.arctan2(-delta[:,0],delta[:,1]),
        np.arctan2(velocity[:,1],velocity[:,0]),delta[:,2],velocity[:,2],yaw),axis=1)

def progress_reward(obs,accel,previous_accel,coeff,speed_threshold=3.):
    d,v,theta,track,dz,vz,yaw=obs.T
    # No angular wrapping here: this is the upstream source's literal behavior.
    terms=np.stack((d,np.where(v>speed_threshold,v,0.),abs(dz),abs(track+yaw-theta),
        np.linalg.norm(accel-previous_accel,axis=1),abs(track)),axis=1)*np.asarray(coeff)
    return terms.sum(1),terms

def preprocess_depth(depth,max_depth=12.):
    # RTX sky is +inf; it represents no hit, hence maximum depth.
    depth=np.nan_to_num(depth,nan=max_depth,posinf=max_depth,neginf=0.)
    return (np.clip(depth,0,max_depth)/max_depth*255.).astype(np.uint8)
