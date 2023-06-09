from task1_forward_kinematics import *
from scipy.spatial.transform import Rotation as R
from Lab2_IK_answers import *
class MetaData:
    def __init__(self, joint_name, joint_parent, joint_initial_position, root_joint, end_joint):
        """
        一些固定信息，其中joint_initial_position是T-pose下的关节位置，可以用于计算关节相互的offset
        root_joint是固定节点的索引，并不是RootJoint节点
        """
        self.joint_name = joint_name
        self.joint_parent = joint_parent
        self.joint_initial_position = joint_initial_position
        self.root_joint = root_joint
        self.end_joint = end_joint

    def get_path_from_root_to_end(self):
        """
        辅助函数，返回从root节点到end节点的路径
        
        输出：
            path: 各个关节的索引
            path_name: 各个关节的名字
        Note: 
            如果root_joint在脚，而end_joint在手，那么此路径会路过RootJoint节点。
            在这个例子下path2返回从脚到根节点的路径，path1返回从根节点到手的路径。
            你可能会需要这两个输出。
        """
        
        # 从end节点开始，一直往上找，直到找到腰部节点
        path1 = [self.joint_name.index(self.end_joint)]
        while self.joint_parent[path1[-1]] != -1:
            path1.append(self.joint_parent[path1[-1]])
            
        # 从root节点开始，一直往上找，直到找到腰部节点
        path2 = [self.joint_name.index(self.root_joint)]
        while self.joint_parent[path2[-1]] != -1:
            path2.append(self.joint_parent[path2[-1]])
        
        # 合并路径，消去重复的节点
        while path1 and path2 and path2[-1] == path1[-1]:
            path1.pop()
            a = path2.pop()
            
        path2.append(a)
        path = path2 + list(reversed(path1))
        path_name = [self.joint_name[i] for i in path]
        return path, path_name, path1, path2
    



def part1_simple(viewer, target_pos):
    """
    完成part1_inverse_kinematics，我们将根节点设在腰部，末端节点设在左手
    """
    viewer.create_marker(target_pos, [1, 0, 0, 1])
    joint_name, joint_parent, joint_initial_position = viewer.get_meta_data()
    meta_data = MetaData(joint_name, joint_parent, joint_initial_position, 'RootJoint', 'lWrist_end')
    joint_position = viewer.get_joint_positions()
    joint_orientation = viewer.get_joint_orientations()
    
    joint_position, joint_orientation = part1_inverse_kinematics(meta_data, joint_position, joint_orientation, target_pos)
    viewer.show_pose(joint_name, joint_position, joint_orientation)
    viewer.run()
    pass


def part1_hard(viewer, target_pos):
    """
    完成part1_inverse_kinematics，我们将根节点设在**左脚部**，末端节点设在左手
    """
    viewer.create_marker(target_pos, [1, 0, 0, 1])
    joint_name, joint_parent, joint_initial_position = viewer.get_meta_data()
    meta_data = MetaData(joint_name, joint_parent, joint_initial_position, 'lToeJoint_end', 'lWrist_end')
    joint_position = viewer.get_joint_positions()
    joint_orientation = viewer.get_joint_orientations()
    
    joint_position, joint_orientation = part1_inverse_kinematics(meta_data, joint_position, joint_orientation, target_pos)
    viewer.show_pose(joint_name, joint_position, joint_orientation)
    viewer.run()
    pass

def part1_animation(viewer, target_pos):
    """
    如果正确完成了part1_inverse_kinematics， 此处不用做任何事情
    可以通过`wasd`控制marker的位置
    """
    marker = viewer.create_marker(target_pos, [1, 0, 0, 1])
    
    joint_name, joint_parent, joint_initial_position = viewer.get_meta_data()
    meta_data = MetaData(joint_name, joint_parent, joint_initial_position, 'lToeJoint_end', 'lWrist_end')
    joint_position = viewer.get_joint_positions()
    joint_orientation = viewer.get_joint_orientations()
    class UpdateHandle:
        def __init__(self, marker, joint_position, joint_orientation):
            self.marker = marker
            self.joint_position = joint_position
            self.joint_orientation = joint_orientation
            
        def update_func(self, viewer):
            target_pos = np.array(self.marker.getPos())
            self.joint_position, self.joint_orientation = part1_inverse_kinematics(meta_data, self.joint_position, self.joint_orientation, target_pos)
            viewer.show_pose(joint_name, self.joint_position, self.joint_orientation)
    handle = UpdateHandle(marker, joint_position, joint_orientation)
    handle.update_func(viewer)
    viewer.update_marker_func = handle.update_func
    viewer.run()


def part2(viewer, bvh_name):
    motion_data = load_motion_data(bvh_name)
    bvh_joint_name, bvh_joint_parent, bvh_offset = part1_calculate_T_pose(bvh_name)
    joint_name, _, joint_initial_position = viewer.get_meta_data()
    idx = [joint_name.index(name) for name in bvh_joint_name]
    meta_data = MetaData(bvh_joint_name, bvh_joint_parent, joint_initial_position[idx], 'lShoulder', 'lWrist')
    class UpdateHandle:
        def __init__(self, meta_data, motion_data, joint_offset):
            self.meta_data = meta_data
            self.motion_data = motion_data
            self.joint_name = meta_data.joint_name
            self.joint_parent = meta_data.joint_parent
            self.joint_offset = joint_offset
            self.current_frame = 0
            
        def update_func(self, viewer):
            joint_position, joint_orientation = part2_forward_kinematics(
                self.joint_name, self.joint_parent, self.joint_offset, self.motion_data, self.current_frame)
            joint_position, joint_orientation = part2_inverse_kinematics(self.meta_data, joint_position, joint_orientation, 0.1, 0.3, 1.4)
            viewer.show_pose(self.joint_name, joint_position, joint_orientation)
            self.current_frame = (self.current_frame + 1) % self.motion_data.shape[0]
    handle = UpdateHandle(meta_data, motion_data, bvh_offset)
    viewer.update_func = handle.update_func
    viewer.run()
    pass

def bonus(viewer, left_target_pos, right_target_pos):
    left_marker = viewer.create_marker(left_target_pos, [1, 0, 0, 1])
    right_marker = viewer.create_marker2(right_target_pos, [0, 0, 1, 1])
    
    joint_name, joint_parent, joint_initial_position = viewer.get_meta_data()
    
    # 为了兼容如此设置，实际上末端节点应当为左右手
    meta_data = MetaData(joint_name, joint_parent, joint_initial_position, 'lToeJoint_end', 'lWrist_end')
    joint_position = viewer.get_joint_positions()
    joint_orientation = viewer.get_joint_orientations()
    
    class UpdateHandle:
        def __init__(self, left_marker, right_marker, joint_position, joint_orientation):
            self.left_marker = left_marker
            self.right_marker = right_marker
            self.joint_position = joint_position
            self.joint_orientation = joint_orientation
            
        def update_func(self, viewer):
            left_target_pos = np.array(self.left_marker.getPos())
            right_target_pos = np.array(self.right_marker.getPos())
            self.joint_position, self.joint_orientation = bonus_inverse_kinematics(meta_data, self.joint_position, self.joint_orientation, left_target_pos, right_target_pos)
            viewer.show_pose(joint_name, self.joint_position, self.joint_orientation)
    handle = UpdateHandle(left_marker, right_marker, joint_position, joint_orientation)
    handle.update_func(viewer)
    
    
    viewer.update_marker_func = handle.update_func
    viewer.run()
    

def main():
    viewer = SimpleViewer()
    
    # part1
    # part1_simple(viewer, np.array([0.5, 0.75, 0.5]))
    # part1_hard(viewer, np.array([0.5, 0.5, 0.5]))
    # part1_animation(viewer, np.array([0.5, 0.5, 0.5]))
    
    # part2
    # part2(viewer, 'data/walk60.bvh')
    
    bonus(viewer, np.array([0.5, 0.5, 0.5]), np.array([0, 0.5, 0.5]))

if __name__ == "__main__":
    main()