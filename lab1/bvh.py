""" BioVision Hierarchy File I/O Functions """
import numpy as np

def load_hierarchy(bvh_file_path, skip_endsite=False, debug=False):
    """ BVH loader: hierarchy """
    hierarchy = {
        "name": [],
        "type": [],
        "parent": [],
        "offset": [],
        "channels": [],
    }
    with open(bvh_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith('HIERARCHY'):
                break
        i += 1
        stack = []
        idx = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('MOTION'):
                break
            if line.startswith('ROOT') or line.startswith('JOINT') or line.startswith('End Site'):
                parent_idx = -1 if len(stack) == 0 else stack[-1]
                if not line.startswith('End Site'):
                    node_type = "ROOT" if line.startswith('ROOT') else "JOINT"
                    name = line[len(node_type) + 1:]
                else:
                    node_type = "EndSite"
                    name = hierarchy["name"][parent_idx] + "_end"
                assert lines[i + 1].strip() == '{'
                i += 1
                if skip_endsite and node_type == "EndSite":
                    stack.append(-2) # dummy index: set -2 for EndSite
                else:
                    stack.append(idx)
                    idx += 1
                line = lines[i + 1].strip()
                offset = []
                if line.startswith('OFFSET'):
                    offset = list(map(float, line.split()[-3:]))
                    i += 1
                line = lines[i + 1].strip()
                channels = []
                if line.startswith('CHANNELS'):
                    channels = line.split()[2:]
                    i += 1
                if debug:
                    print(f"[Name:{name:>16}|Index:{stack[-1]:03d}|Parent:{parent_idx:03d}] >> Type={node_type:<8} | OFFSET = {offset}; CHANNELS = {channels}")
                hierarchy["name"].append(name)
                hierarchy["type"].append(node_type)
                hierarchy["parent"].append(parent_idx)
                hierarchy["offset"].append(offset)
                hierarchy["channels"].append(channels)
            elif line == '}':
                if debug:
                    print(f"[stack-simulation] line#={i:04d}, pop-out:{stack[-1]}")
                stack.pop()
            i += 1
    return hierarchy

def bvh_channels_map(hierarchy):
    """ BVH loader: return slicing width for each channel """
    slice_indices = []
    slice_idx = 0
    for i, channel in enumerate(hierarchy["channels"]):
        step = len(channel)
        slice_indices.append((slice_idx, slice_idx + step))
        slice_idx += step
    return slice_indices

def load_motion_data(bvh_file_path):
    """ BVH loader: motion data """
    with open(bvh_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i + 1: ]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


if __name__ == '__main__':

    # example:
    ch = load_hierarchy("./data/walk60.bvh")
    for k, v in ch.items():
        print(k, '=>', len(v), v)
    
    motion = load_motion_data("./data/walk60.bvh")
    print(motion.shape, motion.dtype)
