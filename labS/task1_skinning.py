'''
一个非常简单的skinning小作业
获得你的蒙皮角色吧！
'''

from Viewer.mesh_viewer import MeshViewer
from answer_task1 import *
import numpy as np



def part1_skinning_one_frame(viewer, translations, orientations, T_pose, skinning_weight, idx, value, frame_id=0):
    translation = translations[frame_id]
    orientation = orientations[frame_id]
    init_abs_position = viewer.init_abs_position
    
    translation = skinning(translation, orientation, T_pose, init_abs_position, idx, value)
    
    viewer.set_vertex_position(translation)
    
    return

def part2_skinning_animation(viewer, translations, orientations, T_pose, skinning_weight, idx, value):
    
    class Animation:
        def __init__(self, *args):
            self.args = args
            self.current_frame = 0
            self.max_frame = len(args[1])
        def update(self, frame_id):
            part1_skinning_one_frame(*self.args, frame_id=self.current_frame)
            self.current_frame = (self.current_frame + 1) % self.max_frame
    
    animation = Animation(viewer, translations, orientations, T_pose, skinning_weight, idx, value)
    viewer.update_func = animation.update

def main():
    viewer = MeshViewer()
    bvh = BVHMotion('motion_material/motion.bvh')
    skinning_weight , name_list, idx, value = viewer.get_skinning_matrix()
    bvh.adjust_joint_name(name_list)
    translations, orientations = bvh.batch_forward_kinematics()
    T_pose = bvh.get_T_pose()
    
    # 请注释掉不需要的部分
    part1_skinning_one_frame(viewer, translations, orientations, T_pose, skinning_weight, idx, value, frame_id=0)    
    # part2_skinning_animation(viewer, translations, orientations, T_pose, skinning_weight, idx, value)
    viewer.run()
    
if __name__ == '__main__':
    main()