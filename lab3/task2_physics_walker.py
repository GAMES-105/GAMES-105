from Viewer.controller import SimpleViewer, Controller
from answer_task2 import *
from physics_warpper import PhysicsInfo, PhysicsHandler, TorqueFunc

def main():
    viewer = SimpleViewer(True)
    from physics_warpper import PhysicsInfo, PhysicsHandler
    physics_info = PhysicsInfo(viewer)
    physics_handler = PhysicsHandler(viewer)
    
    kargs = WalkingController.build_kargs() # 获取初始化参数
    character_controller = WalkingController(physics_info, physics_handler, **kargs)
    
    character_controller.prepare() # 用于训练或优化
    torque_func = TorqueFunc(character_controller.apply_torque, viewer) # 获取力矩策略
    viewer.pre_simulation_func = torque_func.pre_func
    viewer.run()
    
    pass

if __name__ == '__main__':
    main()