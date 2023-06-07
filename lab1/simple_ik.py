""" Simple Iterative IK Solver via CCD and FABRIK """
import numpy as np

NORMALIZE_EPSILON = 1E-9
CCD_ITER_LIMITS = 20
CCD_EUC_EPSILON = 0.005
FABRIK_ITER_LIMITS = 10
FABRIK_EUC_EPSILON = 0.005

def distance(p0, p1):
    """ Euclidean distance """
    return np.linalg.norm(p1 - p0, axis=-1)

def normalize(vec, epsilon=NORMALIZE_EPSILON):
    """ Normalize a vector """
    epsilon = 0.0 if np.linalg.norm(vec) > epsilon else epsilon
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + epsilon)

def rvec_rodrigues(axis, sin_val, cos_val):
    """ Convert axis-angle to rotation matrix """
    e1, e2, e3 = axis
    ex = np.array([
        [0, -e3, e2], 
        [e3, 0, -e1], 
        [-e2, e1, 0],
    ], dtype=np.float64)
    I = np.eye(3, dtype=np.float64)

    A = I * cos_val + (1.0 - cos_val) * axis.reshape((3, 1)) @ axis.reshape((1, 3)) + sin_val * ex
    return A

def compute_rmat_by_vecs(v0, v1):
    """ Rotation_matrix: rotate from v0 to v1 """
    v0 = normalize(v0)
    v1 = normalize(v1)
    # axis-angle
    axis_vec = np.cross(v0, v1)
    axis = normalize(axis_vec)
    sin_val = np.linalg.norm(axis_vec, axis=-1)
    cos_val = v0.reshape((1, 3)) @ v1.reshape((3, 1))
    return rvec_rodrigues(axis, sin_val, cos_val)

def ccd_ik(ps, pt):
    """ 
    Cyclic Coordinate Descent Inverse Kinematics:
    input N points (N-1 links), output N-1 new rotation and N points
    :param ps: points, from root (ik anchor) to end (point to move); shape=(N, 3)
    :param pt: target point; shape=(3,)
    :return: rotation on p0->p1, p1->p2, ..., and points_rotated
    """
    assert ps.shape[-1] == pt.shape[-1]
    rotations = np.repeat(np.eye(3)[None, ...], ps.shape[0] - 1, axis=0)
    for _ in range(CCD_ITER_LIMITS):
        for j in range(ps.shape[0] - 1):
            euc_delta = distance(ps[-1], pt)
            if euc_delta <= CCD_EUC_EPSILON:
                return rotations, ps
            rmat = compute_rmat_by_vecs(v0=ps[-1] - ps[j], v1=pt - ps[j])
            ps[j:, ...] = (ps[j:, ...] - ps[j]) @ rmat.T + ps[j]
            rotations[j] = rotations[j] @ rmat.T
    return rotations, ps

def fabrik(ps, pt):
    """
    Forward And Backward Reaching Inverse Kinematics:
    input N points (N-1 links), output N points
    :param ps: points, from root (ik anchor) to end (point to move); shape=(N, 3)
    :param pt: target point; shape=(3,)
    :return: points_rotated
    """
    assert ps.shape[-1] == pt.shape[-1]
    lens = [distance(ps[i], ps[i + 1]) for i in range(ps.shape[0] - 1)]
    root = ps[0]
    for _ in range(FABRIK_ITER_LIMITS):
        euc_delta = distance(ps[-1], pt)
        if euc_delta <= FABRIK_EUC_EPSILON:
            return ps
        ps[-1] = pt  # forward reaching
        for j in range(1, ps.shape[0]):
            ps[-1 - j] = ps[-j] + normalize(ps[-1 - j] - ps[-j]) * lens[-j]
        ps[0] = root  # backward reaching
        for j in range(1, ps.shape[0]):
            ps[j] = ps[j - 1] + normalize(ps[j] - ps[j - 1]) * lens[j - 1]
    return ps
