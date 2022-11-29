'''
在本任务中，我们会逐步实现一些动作处理的功能，包括
1. 一小段动作的平移与旋转
2. 动作的插值
3. 动作的拼接混合 
4. 动作的循环

拿起纸笔和键盘(?)，开始动手吧！
'''

from Viewer.controller import SimpleViewer
from answer_task1 import *
import numpy as np

class ShowBVHUpdate():
    def __init__(self, viewer, joint_name, translation, orientation):
        self.viewer = viewer
        self.cur_frame = 0
        self.joint_name = joint_name
        self.translation = translation
        self.orientation = orientation
        
    def update(self, task):
        if not self.viewer.update_flag:
            return task.cont
        
        speed_inv = 1 # 控制播放速度的整数,越大越慢
        for i in range(len(self.joint_name)):
            self.viewer.set_joint_position_orientation(self.joint_name[i],
                                                       self.translation[self.cur_frame//speed_inv, i, :],
                                                       self.orientation[self.cur_frame//speed_inv, i, :])
        self.cur_frame = (self.cur_frame + 1) % (self.translation.shape[0]*speed_inv)
        return task.cont
    
def part1_translation_and_rotation(viewer, setting_id):
    
    # 一些不同的设置
    bvh_list = ['motion_material/walk_forward.bvh', 'motion_material/run_forward.bvh', 'motion_material/walk_and_turn_left.bvh']
    pos_xz_list = [np.array([-4,4]), np.array([2,4]), np.array([6,1])]
    facing_xz_list = [np.array([1,1]), np.array([5,1]), np.array([1,1])]
    frame_list = [0, -1, -1]
    
    # 读取设置
    bvh = bvh_list[setting_id]
    pos = pos_xz_list[setting_id]
    facing_xz = facing_xz_list[setting_id]
    frame = frame_list[setting_id]

    original_motion = BVHMotion(bvh)
    new_motion = original_motion.translation_and_rotation(frame, pos, facing_xz)
    
    translation, orientation = new_motion.batch_forward_kinematics()
    task = ShowBVHUpdate(viewer, new_motion.joint_name, translation, orientation) 
    viewer.addTask(task.update)
    
    # 画些参考点
    viewer.create_arrow(np.array([pos[0],1e-3,pos[1]]), facing_xz)
    viewer.create_marker(np.array([pos[0],0,pos[1]]) ,[0,1,0,1])
    return

def part2_interpolate(viewer, v):
    
    # 读取动作
    walk_forward = BVHMotion('motion_material/walk_forward.bvh')
    run_forward = BVHMotion('motion_material/run_forward.bvh')
    run_forward.adjust_joint_name(walk_forward.joint_name)
    
    # 调整方向和位置, 对齐第一帧
    walk_forward = walk_forward.translation_and_rotation(0, np.array([0,0]), np.array([0,1]))
    run_forward = run_forward.translation_and_rotation(0, np.array([0,0]), np.array([0,1]))
    
    # 计算插值系数
    v1 = (walk_forward.joint_position[-1,0,2] / walk_forward.motion_length)*60
    v2 = (run_forward.joint_position[-1,0,2] / run_forward.motion_length)*60
    blend_weight = (v-v1)/(v2-v1)
    distance = (1-blend_weight)*walk_forward.joint_position[-1,0,2] + blend_weight*run_forward.joint_position[-1,0,2]
    cycle_time = np.around(distance / v*60).astype(np.int32)
    alpha = np.ones((cycle_time,)) * blend_weight
    
    # 插值
    motion = blend_two_motions(walk_forward, run_forward, alpha)
    tanslation, orientation = motion.batch_forward_kinematics()
    task = ShowBVHUpdate(viewer, motion.joint_name, tanslation, orientation)
    viewer.addTask(task.update)
    pass

def part3_build_loop(viewer):
    # 不用自己写(但是你可以试着写一下)
    # 推荐阅读 https://theorangeduck.com/
    # Blog名称: Creating Looping Animations from Motion Capture
    motion = BVHMotion('motion_material/run_forward.bvh')
    motion = build_loop_motion(motion)
    
    pos = motion.joint_position[-1,0,[0,2]]
    rot = motion.joint_rotation[-1,0]
    facing_axis = R.from_quat(rot).apply(np.array([0,0,1])).flatten()[[0,2]]
    new_motion = motion.translation_and_rotation(0, pos, facing_axis)
    motion.append(new_motion)
    translation, orientation = motion.batch_forward_kinematics()
    task = ShowBVHUpdate(viewer, motion.joint_name, translation, orientation)
    viewer.addTask(task.update)


def part4_concatenate(viewer, setting_id):
    if setting_id == 0:
        walk_forward = BVHMotion('motion_material/walkF.bvh')
        mix_time = 78 # 一个长motion,手动指定混合时间
    else:
        walk_forward = BVHMotion('motion_material/walk_forward.bvh')
        walk_forward = build_loop_motion(walk_forward)
        mix_time = walk_forward.motion_length # 一个循环motion,自动计算混合时间
        
        # 由于是循环动作,可以把walk_forward直接拼接一遍上去
        motion = walk_forward
        pos = motion.joint_position[-1,0,[0,2]]
        rot = motion.joint_rotation[-1,0]
        facing_axis = R.from_quat(rot).apply(np.array([0,0,1])).flatten()[[0,2]]
        new_motion = motion.translation_and_rotation(0, pos, facing_axis)
        walk_forward.append(new_motion)
    
    run_forward = BVHMotion('motion_material/run_forward.bvh')
    run_forward.adjust_joint_name(walk_forward.joint_name)
    
    motion = concatenate_two_motions(walk_forward, run_forward, mix_time, 30)
    translation, orientation = motion.batch_forward_kinematics()
    task = ShowBVHUpdate(viewer, motion.joint_name, translation, orientation)
    viewer.addTask(task.update)
    pass


def main():
    viewer = SimpleViewer() # 暂时还用不着Controller
    viewer.show_axis_frame()
    
    # 请自行取消需要的注释并更改测试setting_id
    # 请不要同时取消多个注释，否则前者会被后者覆盖
    
    part1_translation_and_rotation(viewer, 0) # 数字代表不同的测试setting
    # part2_interpolate(viewer, 1) # 数字代表不同期望的前进速度
    # part3_build_loop(viewer)
    # part4_concatenate(viewer, 0) # 数字代表不同的测试setting
    viewer.run()
    
if __name__ == '__main__':
    main()