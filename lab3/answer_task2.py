# 以下部分均为可更改部分，但是不要删除必要接口

from answer_task1 import *

class WalkingController():
    
    @staticmethod
    def build_kargs():
        '''
        用于构建初始化的参数字典,比如你希望传入一个motion
        '''
        kargs = {
            'motion': r'motion_material\walkF.bvh'
        }
        return kargs
    
    def __init__(self, physics_info, physics_handler, **kargs) -> None:
        self.physics_info = physics_info
        self.physics_handler = physics_handler
        self.simulated_step = 0
        # 注意必须调整joint_name
        self.motion = BVHMotion(kargs['motion'])
        self.motion.adjust_joint_name(physics_info.joint_name)
    
    def prepare(self):
        '''
        在仿真开始前调用, 可以用于训练或开环优化
        作为实例我们简单地初始化一下姿势
        '''
        translation, orientation = self.motion.batch_forward_kinematics(frame_id_list = [0])
        self.physics_handler.set_pose(translation[0], orientation[0])
        
        pass

    def apply_torque(self):
        '''
        在仿真每一帧调用, 返回值为每个关节应该施加的力矩
        我们这里用的是静态站立力矩
        '''
        torque = part3_cal_static_standing_torque(self.motion, self.physics_info)
        self.simulated_step += 1
        return torque