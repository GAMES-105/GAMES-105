import numpy as np
import math
from scipy.spatial.transform import Rotation as R

def align_quat(qt: np.ndarray, inplace: bool):
    ''' make q_n and q_n+1 in the same semisphere
        the first axis of qt should be the time
    '''
    qt = np.asarray(qt)
    if qt.shape[-1] != 4:
        raise ValueError('qt has to be an array of quaterions')

    if not inplace:
        qt = qt.copy()

    if qt.size == 4:  # do nothing since there is only one quation
        return qt

    sign = np.sum(qt[:-1] * qt[1:], axis=-1)
    sign[sign < 0] = -1
    sign[sign >= 0] = 1
    sign = np.cumprod(sign, axis=0, )

    qt[1:][sign < 0] *= -1
    return qt

def quat_to_avel(rot, dt):
    '''
    用有限差分计算角速度, 假设第一维度是时间
    '''
    rot = align_quat(rot, inplace=False)
    quat_diff = (rot[1:] - rot[:-1])/dt
    quat_diff[...,-1] = (1 - np.sum(quat_diff[...,:-1]**2, axis=-1)).clip(min = 0)**0.5
    quat_tmp = rot[:-1].copy()
    quat_tmp[...,:3] *= -1
    shape = quat_diff.shape[:-1]
    rot_tmp = R.from_quat(quat_tmp.reshape(-1, 4)) * R.from_quat(quat_diff.reshape(-1, 4))
    return 2 * rot_tmp.as_quat().reshape( shape + (4, ) )[...,:3]

def halflife2dampling(halflife):
    return 4 * math.log(2) / halflife

def decay_spring_implicit_damping_pos(pos, vel, halflife, dt):
    '''
    一个阻尼弹簧, 用来衰减位置
    '''
    d = halflife2dampling(halflife)/2
    j1 = vel + d * pos
    eydt = math.exp(-d * dt)
    pos = eydt * (pos+j1*dt)
    vel = eydt * (vel - j1 * dt * d)
    return pos, vel

def decay_spring_implicit_damping_rot(rot, avel, halflife, dt):
    '''
    一个阻尼弹簧, 用来衰减旋转
    '''
    d = halflife2dampling(halflife)/2
    j0 = rot
    j1 = avel + d * j0
    eydt = math.exp(-d * dt)
    a1 = eydt * (j0+j1*dt)
    
    rot_res = R.from_rotvec(a1).as_rotvec()
    avel_res = eydt * (avel - j1 * dt * d)
    return rot_res, avel_res

def build_loop_motion(bvh_motion, half_life = 0.2, fps = 60):
    
    # ---------------处理rotations----------------#
    rotations = bvh_motion.joint_rotation
    avel = quat_to_avel(rotations, 1/60)
    
    # 计算最后一帧和第一帧的旋转差
    rot_diff = (R.from_quat(rotations[-1]) * R.from_quat(rotations[0].copy()).inv()).as_rotvec()
    avel_diff = (avel[-1] - avel[0])
    
    # 将旋转差均匀分布到每一帧
    for i in range(bvh_motion.motion_length):
        offset1 = decay_spring_implicit_damping_rot(
            0.5*rot_diff, 0.5*avel_diff, half_life, i/fps
        )
        offset2 = decay_spring_implicit_damping_rot(
            -0.5*rot_diff, -0.5*avel_diff, half_life, (bvh_motion.motion_length-i-1)/fps
        )
        offset_rot = R.from_rotvec(offset1[0] + offset2[0])
        bvh_motion.joint_rotation[i] = (offset_rot * R.from_quat(rotations[i])).as_quat() 
    
    # -------------------处理positions------------------#
    
    pos_diff = bvh_motion.joint_position[-1] - bvh_motion.joint_position[0]
    pos_diff[:,[0,2]] = 0
    vel1 = bvh_motion.joint_position[-1] - bvh_motion.joint_position[-2]
    vel2 = bvh_motion.joint_position[1] - bvh_motion.joint_position[0]
    vel_diff = (vel1 - vel2)/60
    
    for i in range(bvh_motion.motion_length):
        offset1 = decay_spring_implicit_damping_pos(
            0.5*pos_diff, 0.5*vel_diff, half_life, i/fps
        )
        offset2 = decay_spring_implicit_damping_pos(
            -0.5*pos_diff, -0.5*vel_diff, half_life, (bvh_motion.motion_length-i-1)/fps
        )
        offset_pos = offset1[0] + offset2[0]
        bvh_motion.joint_position[i] += offset_pos
    
    return bvh_motion