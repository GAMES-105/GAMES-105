from Viewer.controller import SimpleViewer, Controller
from answer_task2 import *

def main():
    viewer = SimpleViewer(True, 16)
    from physics_warpper import PhysicsInfo, PhysicsHandler
    physics_info = PhysicsInfo(viewer)
    physics_handler = PhysicsHandler(viewer)
    
    kargs = WalkingController.build_kargs()
    character_controller = WalkingController(physics_info, physics_handler, **kargs)
    # character_controller.prepare()
    state = np.load('states.npy', allow_pickle=True)
    class updater():
        def __init__(self, state):
            self.state = state
            self.cnt = 0
        def update(self, viewer):
            physics_handler.set_state(self.state[self.cnt])
            self.cnt = (self.cnt + 1) % len(self.state)
    viewer.update_func = updater(state).update
    viewer.pre_simulation_func = character_controller.apply_torque
    
    viewer.run()
    pass

if __name__ == '__main__':
    main()