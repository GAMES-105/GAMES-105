import numpy as np
from scipy.spatial.transform import Rotation as R

from quaternion import quat_normalize, hamilton_product, conjugate, rotate
from jacobian_ik import jacobian_transpose, jacobian_pseudo_inverse, jacobian_damped_least_squares
jacobian_method = jacobian_transpose
n_iter_limit = 20

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path, _, _, _ = meta_data.get_path_from_root_to_end()
    joint_positions = joint_positions.copy()
    joint_orientations = joint_orientations.copy()

    # step-1: compute locals
    parents = meta_data.joint_parent
    offsets = [
        meta_data.joint_initial_position[idx] - (meta_data.joint_initial_position[par_idx] if par_idx != -1 else np.array([0, 0, 0]))
        for idx, par_idx in enumerate(parents)
    ]
    local_rotations = np.array([
        quat_normalize(hamilton_product(conjugate(joint_orientations[par_idx]), joint_orientations[idx])) if par_idx != -1 else joint_orientations[idx]
        for idx, par_idx in enumerate(parents)
    ])

    # step-2: ik chain
    glb_pos = joint_positions[path].copy()
    glb_rot = joint_orientations[path[:-1]].copy()
    _ = jacobian_method(
        gp=glb_pos, gr=glb_rot, t=target_pose,
        use_left_perturbation=False,
        n_iter=n_iter_limit)
    root_pos = joint_positions[meta_data.joint_name.index("RootJoint")].copy()
    for i, idx in enumerate(path[:-1]):
        par_idx = parents[idx]
        if par_idx == -1:
            local_rotations[idx] = glb_rot[i]
            root_pos = glb_pos[i]
        else:
            local_rotations[idx] = \
                quat_normalize(hamilton_product(
                    conjugate(glb_rot[path.index(par_idx)] if par_idx in path[:-1] else joint_orientations[par_idx]),
                    glb_rot[i]
                ))

    # step-3: apply (fk)
    for idx, offset in enumerate(offsets):
        par_idx = parents[idx]
        if par_idx == -1:
            joint_positions[idx] = root_pos
            joint_orientations[idx] = local_rotations[idx]
        else:
            joint_positions[idx] = joint_positions[par_idx] + rotate(joint_orientations[par_idx], offset)
            joint_orientations[idx] = hamilton_product(joint_orientations[par_idx], local_rotations[idx])

    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    abs_target = joint_positions[meta_data.joint_name.index("RootJoint")] + np.array([relative_x, 0.0, relative_z])
    abs_target[1] = target_height
    joint_positions, joint_orientations = part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, abs_target)
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    # print(meta_data.get_path_from_root_to_end())
    meta_chain1 = type(meta_data)(meta_data.joint_name, meta_data.joint_parent, meta_data.joint_initial_position, 'lToeJoint_end', 'lWrist_end')
    meta_chain2 = type(meta_data)(meta_data.joint_name, meta_data.joint_parent, meta_data.joint_initial_position, 'lToeJoint_end', 'rWrist_end')

    joint_positions = joint_positions.copy()
    joint_orientations = joint_orientations.copy()
    # brutal force iteration
    for i in range(30):
        joint_positions, joint_orientations = \
            part1_inverse_kinematics(meta_chain1, joint_positions, joint_orientations, left_target_pose)
        joint_positions, joint_orientations = \
            part1_inverse_kinematics(meta_chain2, joint_positions, joint_orientations, right_target_pose)
        err_chain1 = np.linalg.norm(joint_positions[meta_chain1.joint_name.index("lWrist_end")] - left_target_pose)
        err_chain2 = np.linalg.norm(joint_positions[meta_chain2.joint_name.index("rWrist_end")] - right_target_pose)
        print(f"[Iteration:{i:02d}] err_1 = {err_chain1}, err_2 = {err_chain2}")
        if err_chain1 < 1E-4 and err_chain2 < 1E-4: # early-stop
            break

    return joint_positions, joint_orientations