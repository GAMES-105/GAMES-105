import numpy as np
from scipy.spatial.transform import Rotation as R
from bvh import load_hierarchy, load_motion_data, bvh_channels_map
from quaternion import hamilton_product, conjugate, rotate

def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []
    character = load_hierarchy(bvh_file_path)
    for i, _ in enumerate(character["type"]):
        joint_name.append(character["name"][i])
        joint_parent.append(character["parent"][i])
        joint_offset.append(character["offset"][i])
    joint_offset = np.array(joint_offset)
    return joint_name, joint_parent, joint_offset

def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = []
    joint_orientations = []
    motion = motion_data[frame_id] # .shape = (#channels)
    slice_idx = 0
    for i, offset in enumerate(joint_offset):
        par_idx = joint_parent[i]
        if i == 0: # joint_name[i] == 'RootJoint'
            data_slice = motion[slice_idx: slice_idx + 6]
            slice_idx += 6
            position = data_slice[:3] # + np.array(offset)
            orientation = R.from_euler('XYZ', data_slice[3:], True).as_quat()
        elif joint_name[i].endswith('_end'):
            position = joint_positions[par_idx] + rotate(joint_orientations[par_idx], np.array(offset), seq='xyzw')
            orientation = np.array([0, 0, 0, 1]) # dummy: set identity
        else:
            data_slice = motion[slice_idx: slice_idx + 3]
            slice_idx += 3
            rotation = R.from_euler('XYZ', data_slice, True).as_quat()
            position = joint_positions[par_idx] + rotate(joint_orientations[par_idx], np.array(offset), seq='xyzw')
            orientation = hamilton_product(joint_orientations[par_idx], rotation, seq='xyzw')
        joint_positions.append(position)
        joint_orientations.append(orientation)
    joint_positions = np.array(joint_positions)
    joint_orientations = np.array(joint_orientations)
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = []
    # load characters
    ch_T = load_hierarchy(T_pose_bvh_path, debug=False)
    ch_A = load_hierarchy(A_pose_bvh_path, debug=False)
    mo = load_motion_data(A_pose_bvh_path)
    cmap = bvh_channels_map(ch_A)
    # print(set(ch_T["name"]) == set(ch_A["name"])) # True
    glbOrientOffset = {
        "lShoulder": R.from_euler('XYZ', [[0, 0, +45]], True),
        "rShoulder": R.from_euler('XYZ', [[0, 0, -45]], True),
    } # retarget: local rotation convert: TPose to APose
    ### global (orientation) should be the same
    mo_ret = []
    for i, name in enumerate(ch_T["name"]):
        j = ch_A["name"].index(name)
        s_st, s_ed = cmap[j]
        if j == 0: # joint_name[i] == 'RootJoint'
            mo_ret.append(mo[:, s_st: s_ed])
        elif name.endswith('_end'):
            assert s_st == s_ed # dummy end site
        else:
            if name in glbOrientOffset:
                goo = glbOrientOffset[name]
                mo_ret.append((R.from_euler('XYZ', mo[:, s_st: s_ed], True) * goo.inv()).as_euler('XYZ', True))
            else:
                mo_ret.append(mo[:, s_st: s_ed])
    motion_data = np.concatenate(mo_ret, axis=-1)
    return motion_data
