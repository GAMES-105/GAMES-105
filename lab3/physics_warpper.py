import numpy as np

class TorqueFunc():
    def __init__(self, torque_func, viewer) -> None:
        self.torque_func = torque_func
        self.viewer = viewer
        
    def pre_func(self):
        torque = self.torque_func()
        # torque = np.zeros((len(self.viewer.joint_name), 3))
        torque[0] = np.zeros_like(torque[0])
        self.viewer.set_torque(torque)
                
class PhysicsInfo():
    '''
    用于获取viewer当前的物理信息
    '''
    def __init__(self, viewer) -> None:
        self.__viewer = viewer
    
    @property
    def joint_name(self):
        return self.__viewer.joint_name
    
    @property
    def parent_index(self):
        return self.__viewer.parent_index
    
    @property
    def root_idx(self):
        return 0
    
    @property
    def substep(self):
        return self.__viewer.substep
    
    @property
    def root_pos(self):
        return self.__viewer.root_pos
    
    @property
    def root_quat(self):
        return self.__viewer.root_quat
    
    def get_root_pos_and_vel(self):
        return self.__viewer.get_root_pos_vel()
    
    def get_joint_translation(self):
        return self.__viewer.get_physics_joint_positions()
    
    def get_joint_orientation(self):
        return self.__viewer.get_physics_joint_orientations()
    
    def get_body_position(self):
        return self.__viewer.get_physics_body_positions()
    
    def get_body_velocity(self):
        return self.__viewer.get_body_velocities()
    
    def get_body_angular_velocity(self):
        return self.__viewer.get_body_angular_velocities()
    
    def get_body_mass(self):
        '''
        为了让仿真稳定一些，mass的值要比实际方块的大...
        '''
        return self.__viewer.get_body_mass()
    
class PhysicsHandler():
    '''
    主要用于set pose
    '''
    def __init__(self, viewer) -> None:
        self.viewer = viewer
    
    def get_state(self):
        '''
        获取当前的状态，用于保存
        '''
        viewer = self.viewer
        return {
            'velocity': viewer.get_body_velocities(),
            'angular_velocity': viewer.get_body_angular_velocities(),
            'joint_position': viewer.get_physics_joint_positions(),
            'joint_orientation': viewer.get_physics_joint_orientations(),
            'body_position': viewer.get_physics_body_positions(),
        }
        
    def set_state(self, state_dict):
        self.set_physics_joints(state_dict['joint_position'], state_dict['joint_orientation'])
        self.viewer.set_body_velocities(state_dict['velocity'])
        self.viewer.set_body_angular_velocities(state_dict['angular_velocity'])
    
    def get_pose(self):
        return self.viewer.get_pose()
    
    def set_pose(self, joint_translation, joint_orientation):
        self.viewer.set_physics_joints(joint_translation, joint_orientation)
    
    
    def sync_to_kinematics(self):
        self.viewer.sync_physics_to_kinematics()
    
    def simulate(self, torque_func, **kargs):
        
        
        torque_func = TorqueFunc(torque_func, self.viewer)
        
        self.viewer.simulationTask(torque_func.pre_func, **kargs)