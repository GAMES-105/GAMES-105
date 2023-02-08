from Viewer.viewer import SimpleViewer
from bvh_loader import BVHMotion
import numpy as np
from answer_task1_copy import *
from physics_warpper import PhysicsInfo
class PDController:
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.physics_info = PhysicsInfo(viewer)
        self.cnt = 0
        self.get_pose = None
        pass
    
    def apply_pd_torque(self):
        pose = self.get_pose(self.cnt)
        torque = part1_cal_torque(pose, self.physics_info)
        torque[0] = np.zeros_like(torque[0])
        self.viewer.set_torque(torque)
        self.cnt += 1

    def apply_root_force_and_torque(self):
        position, pose, setting = self.get_pose(self.cnt)
        global_force, torque = part2_cal_float_base_torque(position[0], pose, self.physics_info
                                                           )
        if setting == 0:
            torque[0] = np.zeros_like(torque[0])
        self.viewer.set_torque(torque)
        self.viewer.set_root_force(global_force)
        self.cnt += 1
    
    def apply_static_torque(self):
        motion = self.get_pose(self.cnt)
        torque = part3_cal_static_standing_torque(motion, self.physics_info)
        torque[0] = np.zeros_like(torque[0])
        self.viewer.set_torque(torque)
        self.cnt += 1
        
def part1_pd_control(viewer, setting=0):
    
    motion_list = [r'motion_material\physics_motion\long_walk.bvh', r"motion_material\idle.bvh", r"motion_material\walkF.bvh"]
    
    motion = BVHMotion(motion_list[setting])
    motion.adjust_joint_name(viewer.joint_name)
    
    # 计算下姿势， 由于我们把根节点固定了，所以要把根节点的位置也传进去 避免违反约束
    joint_translation, joint_orientation = motion.batch_forward_kinematics(frame_id_list = [0],
                                                                           root_pos = viewer.root_pos,
                                                                           )
    viewer.set_pose(motion.joint_name ,joint_translation[0], joint_orientation[0])
    
    pd_controller = PDController(viewer)
    pd_controller.get_pose = lambda x: motion.joint_rotation[0]
    viewer.pre_simulation_func = pd_controller.apply_pd_torque
    pass   

def part2_root_force(viewer, setting=0):
    '''
    setting 0: track 第一帧
    setting 1: track 整个动作
    '''
    
    motion_list = [r"motion_material\physics_motion\long_walk.bvh"]
    motion = BVHMotion(motion_list[0])
    motion.adjust_joint_name(viewer.joint_name)
    pos = viewer.root_pos
    pos[1] = motion.joint_position[0][0][1] 
    if setting == 1:
        pos[1] += 0.12 #我们把人物往上提亿点
    motion = motion.translation(0, pos)
    joint_translation, joint_orientation = motion.batch_forward_kinematics(frame_id_list = [0],
                                                                           )
    viewer.set_pose(motion.joint_name ,joint_translation[0], joint_orientation[0])
    
    pd_controller = PDController(viewer)
    idx_map = lambda x: x//viewer.substep if setting == 1 else 0
    pd_controller.get_pose = lambda x: (motion.joint_position[idx_map(x)], motion.joint_rotation[idx_map(x)], setting)
    viewer.pre_simulation_func = pd_controller.apply_root_force_and_torque
    pass

def part3_static_balance(viewer, setting):
    motion_list = [r"motion_material\physics_motion\long_walk.bvh"]
    motion = BVHMotion(motion_list[setting])
    motion.adjust_joint_name(viewer.joint_name)
    pos = viewer.root_pos
    pos[1] = motion.joint_position[0][0][1]
    motion = motion.translation(0, pos)
    joint_translation, joint_orientation = motion.batch_forward_kinematics(frame_id_list = [0],
                                                                           )
    viewer.set_pose(motion.joint_name ,joint_translation[0], joint_orientation[0])
    
    pd_controller = PDController(viewer)
    pd_controller.get_pose = lambda x: motion
    viewer.pre_simulation_func = pd_controller.apply_static_torque
    pass

def main():
    viewer = SimpleViewer(True) 
    # viewer.show_axis_frame()
    
    part1_pd_control(viewer, 0) # 数字代表不同的测试setting
    # part2_root_force(viewer, 0)
    # part3_static_balance(viewer, 0)
    viewer.run()
    
if __name__ == '__main__':
    main()