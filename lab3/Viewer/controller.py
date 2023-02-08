'''
This code is highly inspired by the code from the following link:
https://github.com/orangeduck/Motion-Matching/blob/main/controller.cpp
'''

from .viewer import SimpleViewer
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from .visualize_utils import *
from scipy.spatial.transform import Slerp
from panda3d.core import *
from direct.showbase.ShowBase import ShowBase

class KeyAndPad():
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.is_down = viewer.mouseWatcherNode.is_button_down
        self.use_gamepad = False
        self.input_vel = np.zeros(3)
        try:
            self.device = viewer.devices.getDevices(InputDevice.DeviceClass.gamepad)[0]
        except IndexError:
            self.device = None
            
        if self.device:
            self.use_gamepad = True
            self.set_gamepad_map()
        else:
            self.set_key_map()
        
        self.gait = 0
        
    def key_input(self, axis, value):
        if axis == 'x':
            self.input_vel[0] = value
        elif axis == 'z':
            self.input_vel[2] = value
        elif axis == 'gait':
            self.gait = value
            
    def set_gamepad_map(self):
        self.gamepad_map = {
            'x': InputDevice.Axis.left_x,
            'z': InputDevice.Axis.left_y,
        }
        self.viewer.attachInputDevice(self.device, prefix="gamepad")
        self.viewer.taskMgr.add(self.update_gamepad, 'update_gamepad', sort = 1)
        self.viewer.accept('gamepad-rshoulder', self.key_input, ['gait', 1])
        self.viewer.accept('gamepad-rshoulder-up', self.key_input, ['gait', 0])
        
    def update_gamepad(self, task):
        self.input_vel[0] = -self.device.findAxis(self.gamepad_map['x']).value
        self.input_vel[2] = self.device.findAxis(self.gamepad_map['z']).value
        if np.linalg.norm(self.input_vel) > 1:
            self.input_vel /= np.linalg.norm(self.input_vel)
        elif np.linalg.norm(self.input_vel) < 0.2:
            # a dead zone
            self.input_vel = np.zeros(3)
        
        right_x = self.device.findAxis(InputDevice.Axis.right_x).value
        right_y = self.device.findAxis(InputDevice.Axis.right_y).value
        self.viewer.cameractrl.updateGamepad(right_x, right_y, task)
        return task.cont
    
    def set_key_map(self):
        key_map = {
            ('w', 'arrow_up'): ['z', 1],
            ('s', 'arrow_down'): ['z', -1],
            ('a', 'arrow_left'): ['x', 1],
            ('d', 'arrow_right'): ['x', -1],
            ('space', 'space'): ['gait', 1]
        }
        for key, value in key_map.items():
            self.viewer.accept(key[0], self.key_input, [value[0], value[1]])
            self.viewer.accept(key[0] + '-up', self.key_input, [value[0], 0])
            self.viewer.accept(key[1], self.key_input, [value[0], value[1]])
            self.viewer.accept(key[1] + '-up', self.key_input, [value[0], 0])

    def get_input(self):
        return self.input_vel
    
def from_euler(e):
    return R.from_euler('XYZ', e, degrees=True)

class InterpolationHelper():
    @staticmethod
    def lerp(a, b, t):
        return a + (b - a) * t
    
    @staticmethod
    def halflife2dampling(halflife):
        return 4 * math.log(2) / halflife

    def simulation_positions_update(pos, vel, acc, target_vel, halflife, dt):
        d = InterpolationHelper.halflife2dampling(halflife)/2
        j0 = vel - target_vel
        j1 = acc + d * j0
        eydt = math.exp(-d * dt)
        pos_prev = pos
        tmp1 = j0+j1*dt
        tmp2 = j1/(d*d)
        pos = eydt * ( -tmp2 -tmp1/d ) + tmp2 + j0/d + target_vel*dt + pos_prev
        vel = eydt*tmp1 + target_vel
        acc = eydt * (acc - j1*d*dt)
        return pos, vel, acc
    @staticmethod
    def simulation_rotations_update(rot, avel, target_rot, halflife, dt):
        d = InterpolationHelper.halflife2dampling(halflife)/2
        j0 = R.from_quat(rot) * R.from_quat(target_rot).inv()
        j0 = j0.as_rotvec()
        j1 = avel + d * j0
        eydt = math.exp(-d * dt)
        tmp1 = eydt * (j0 + j1 * dt)
        rot = R.from_rotvec(tmp1) * R.from_quat(target_rot)
        rot = rot.as_quat()
        avel = eydt * (avel - j1 * dt * d)
        return rot, avel
    
    @staticmethod
    def decay_spring_implicit_damping_rot(rot, avel, halflife, dt):
        d = InterpolationHelper.halflife2dampling(halflife)/2
        j0 = from_euler(rot).as_rotvec()
        j1 = avel + d * j0
        eydt = math.exp(-d * dt)
        a1 = eydt * (j0+j1*dt)
       
        rot_res = R.from_rotvec(a1).as_euler('XYZ', degrees=True)
        avel_res = eydt * (avel - j1 * dt * d)
        return rot_res, avel_res
    
    @staticmethod
    def decay_spring_implicit_damping_pos(pos, vel, halflife, dt):
        d = InterpolationHelper.halflife2dampling(halflife)/2
        j1 = vel + d * pos
        eydt = math.exp(-d * dt)
        pos = eydt * (pos+j1*dt)
        vel = eydt * (vel - j1 * dt * d)
        return pos, vel
    
    @staticmethod
    def inertialize_transition_rot(prev_off_rot, prev_off_avel, src_rot, src_avel, dst_rot, dst_avel):
        prev_off_rot, prev_off_avel = InterpolationHelper.decay_spring_implicit_damping_rot(prev_off_rot, prev_off_avel, 1/20, 1/60)
        off_rot = from_euler(prev_off_rot) * from_euler(src_rot) * from_euler(dst_rot).inv()
        off_avel = prev_off_avel + src_avel - dst_avel
        # off_rot = from_euler(src_rot) * from_euler(dst_rot).inv()
        # off_avel = src_avel - dst_avel
        return off_rot.as_euler('XYZ', degrees=True), off_avel
    
    @staticmethod
    def inertialize_update_rot(prev_off_rot, prev_off_avel, rot, avel, halflife, dt):
        off_rot , off_avel = InterpolationHelper.decay_spring_implicit_damping_rot(prev_off_rot, prev_off_avel, halflife, dt)
        rot = from_euler(off_rot) * from_euler(rot)
        avel = off_avel + avel
        return rot.as_euler('XYZ', degrees=True), avel, off_rot, off_avel
    
    @staticmethod
    def inertialize_transition_pos(prev_off_pos, prev_off_vel, src_pos, src_vel, dst_pos, dst_vel):
        prev_off_pos, prev_off_vel = InterpolationHelper.decay_spring_implicit_damping_pos(prev_off_pos, prev_off_vel, 1/20, 1/60)
        off_pos = prev_off_pos + src_pos - dst_pos
        off_vel = prev_off_vel + src_vel - dst_vel
        return off_pos, off_vel
    
    @staticmethod
    def inertialize_update_pos(prev_off_pos, prev_off_vel, pos, vel, halflife, dt):
        off_pos , off_vel = InterpolationHelper.decay_spring_implicit_damping_pos(prev_off_pos, prev_off_vel, halflife, dt)
        pos = off_pos + pos
        vel = off_vel + vel
        return pos, vel, off_pos, off_vel
    
class Controller:
    def __init__(self, viewer) -> None:
        
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        self.avel = np.zeros(3)
        self.desired_rotation = np.array([0,0,0,1])

        self.dt = 1/60
        self.viewer = viewer
        viewer.taskMgr.add(self.update)
        self.future_step = 6
        self.futures = []
        
        self.future_vel = []
        self.future_avel = []
        self.future_pos = []
        self.future_rot = []
        
        self.desired_velocity_change = np.zeros(3)
        self.desired_rotation_change = np.zeros(3)
        
        for i in range(self.future_step):
            node = self.viewer.render.attach_new_node('future{i}')
            node.setPos(0,0.01,0)
            draw_circle_with_arrow(node, 0.5, (252/255, 173/255, 5/155,1), with_circle = i == 0)
            node.reparentTo(self.viewer.render)
            self.futures.append(node)
        self._node = self.futures[0]
        self.init_key_input()
        self.halflife = 0.27
        self.move_speed = np.array([1.75, 1.5, 1.25])
    @property
    def node(self):
        return self._node
    @property
    def rotation(self):
        return np.array(self.node.get_quat())[[1,2,3,0]]
    @property
    def cameractrl(self):
        return self.viewer.cameractrl
    @property
    def input_vel(self):
        return self.input_device.get_input()

    
    def desired_velocity_update(self, camera_to_pos, input_vel, simulation_rotation):
        camera_to_pos[1] = 0
        
        fwrd_speed, side_speed, back_speed = self.move_speed
        
        angle = np.arctan2(camera_to_pos[0], camera_to_pos[2])
        rot = R.from_rotvec( angle * np.array([0,1,0]) )
        global_direction = rot.apply(input_vel)
        
        local_desired_direction = R.from_quat(simulation_rotation).inv().apply(global_direction)
        local_desired_velocity = np.array([side_speed, 0, fwrd_speed]) * local_desired_direction \
            if local_desired_direction[2] > 0 else np.array([side_speed, 0, back_speed]) * local_desired_direction
        
        desired_velocity = R.from_quat(simulation_rotation).apply(local_desired_velocity)
        return desired_velocity
    
    def desired_rotation_update(self, cur_rotation, desired_velocity):
        if np.linalg.norm(desired_velocity) < 1e-5:
            # return cur_rotation
            return self.rotation
        else:
            desired_direction = desired_velocity / np.linalg.norm(desired_velocity)
            return R.from_rotvec( np.arctan2(desired_direction[0], desired_direction[2]) * np.array([0,1,0]) ).as_quat()
    
    def init_key_input(self):
        self.input_device = KeyAndPad(self.viewer)
        
        node = self.node.attach_new_node('camera_pos')
        node.setPos(0, 3, -5)
        self.cameractrl.position = node.getPos(self.viewer.render)
        self.camera_ref_pos = node
        self.camera_ref_pos.wrtReparentTo(self.node)
        self.line = LineSegs()
        self.geom = self.line.create(dynamic = True)
        self.viewer.render.attach_new_node(self.geom)
    
    @property
    def current_desired_rotation(self):
        return np.array(self.futures[0].get_quat())[[1,2,3,0]]
    
    @property
    def current_desired_position(self):
        return np.array(self.futures[0].get_pos())
    
    @property
    def gait(self):
        return self.input_device.gait
    
    def update_pos(self):
        # self.input_vel = np.array([1,0,0])
        init_pos = self.node.get_pos()
        init_rot = self.rotation
        self.sub_step = 20
        
        # map input to desired velocity
        camera_fwd = self.cameractrl.center - self.cameractrl.position
        cur_target_vel = self.desired_velocity_update(camera_fwd, self.input_vel, init_rot)
        cur_target_rot = self.desired_rotation_update(self.desired_rotation, cur_target_vel)
        self.desired_rotation = cur_target_rot
        self.desired_vel = cur_target_vel
        
        self.desired_velocity_change = (cur_target_vel - self.vel)/self.dt
        self.desired_rotation_change = (R.from_quat(cur_target_rot).inv() * R.from_quat(init_rot)).as_rotvec()/self.dt
        
        # predict future rotations
        rotation_trajectory = [init_rot]
        new_rot, new_avel = init_rot, self.avel
        self.future_avel = [new_avel]
        for i in range(self.future_step):
            new_rot, new_avel = InterpolationHelper.simulation_rotations_update(new_rot, new_avel, cur_target_rot, self.halflife, self.dt* self.sub_step )
            rotation_trajectory.append(new_rot)
            self.future_avel.append(new_avel.copy())
        
        # predict future positions
        new_pos, new_vel, new_acc = init_pos, self.vel, self.acc
        position_trajectory = [init_pos]
        self.future_vel = [new_vel]
        for i in range(self.future_step - 1):
            new_pos, new_vel, new_acc = InterpolationHelper.simulation_positions_update(new_pos, new_vel, new_acc, cur_target_vel, self.halflife, self.dt* self.sub_step )
            position_trajectory.append(new_pos)
            self.future_vel.append(new_vel.copy())
           
        # update current positions
        rotation_trajectory[0], self.avel = InterpolationHelper.simulation_rotations_update(init_rot, self.avel, cur_target_rot, self.halflife, self.dt)
        position_trajectory[0], self.vel, self.acc = InterpolationHelper.simulation_positions_update(init_pos, self.vel, self.acc, cur_target_vel, self.halflife, self.dt)
        rotation_trajectory = np.array(rotation_trajectory).reshape(-1, 4)
        
        # record the trajectory
        self.future_pos = np.array(position_trajectory).reshape(-1, 3)
        self.future_rot = rotation_trajectory.copy()
        self.future_vel[0] = self.vel.copy()
        self.future_vel = np.array(self.future_vel).reshape(-1, 3)
        self.future_avel[0] = self.avel.copy()
        self.future_avel = np.array(self.future_avel).reshape(-1, 3)
        
        rotation_trajectory = rotation_trajectory[...,[3,0,1,2]]
        for i in range(self.future_step):
            self.futures[i].set_pos(*position_trajectory[i])
            self.futures[i].set_quat( Quat(*rotation_trajectory[i]) )
        
        # update camera positions
        delta = position_trajectory[0] - init_pos
        delta = LVector3(*delta)
        self.cameractrl.position = self.cameractrl.position + delta
        self.cameractrl.center = self.cameractrl.center + delta
        self.cameractrl.look()
        
    def draw_future(self):
        self.line.reset()
        self.line.set_color(240/255,31/255,141/255,0.1)
        self.line.setThickness(3)
        positions = [np.array(self.futures[i].get_pos()) for i in range(self.future_step)]
        self.line.moveTo(*positions[0])
        for i in positions[1:]:
            self.line.drawTo(i[0], i[1], i[2])
        self.geom.remove_all_geoms()
        self.line.create(self.geom, True)
    
    def update(self, task):
        self.update_pos()
        self.draw_future()
        return task.cont
    
    def set_pos(self, pos):
        
        init_pos = self.node.get_pos()
        pos = pos.copy()
        pos[1] = 0.01
        self.node.set_pos(*pos)

        delta = pos - init_pos
        delta = LVector3(*delta)
        self.cameractrl.position = self.cameractrl.position + delta
        self.cameractrl.center = self.cameractrl.center + delta
        self.cameractrl.look()
        
        
    def set_rot(self, rot):
        rot = rot.copy()
        facing = R.from_quat(rot).apply(np.array([0,0,1]))
        facing_xz = np.array([facing[0], 0, facing[2]])  
        rot = R.from_rotvec( np.arctan2(facing_xz[0], facing_xz[2]) * np.array([0,1,0]) ).as_quat()
        self.node.set_quat(Quat(rot[3], rot[0], rot[1], rot[2]))
    
    def get_desired_state(self):
        return self.future_pos, self.future_rot, self.future_vel, self.future_avel, self.gait
    
def main():
    viewer = SimpleViewer()
    viewer.show_axis_frame()
    controller = Controller(viewer)
    viewer.run()


if __name__=="__main__":
    main()
    