from VclSimuBackend.pymotionlib import BVHLoader
import numpy as np
bvh = BVHLoader.load(r'motion_material\physics_motion\long_run.bvh')
bvh = bvh.sub_sequence(2170, 2215)
# bvh = bvh.resample(60)
# bvh = bvh.flip(np.array([1,0,0]))
BVHLoader.save(bvh, r'motion_material\run_forward.bvh')