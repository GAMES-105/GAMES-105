""" Quaternion Algebra """
import numpy as np

QUAT_EPSILON = 1E-16

def quat_normalize(q):
    return q / (np.linalg.norm(q) + QUAT_EPSILON)

def hamilton_product(q1, q2, seq='xyzw'):
    """ quaternion product """
    assert seq in ('wxyz', 'xyzw')
    if seq == 'wxyz':
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
    else:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    if seq == 'wxyz':
        return np.array([w, x, y, z])
    else:
        return np.array([x, y, z, w])

def conjugate(q, seq='xyzw'):
    """ quaternion inverse """
    assert seq in ('wxyz', 'xyzw')
    if seq == 'wxyz':
        return np.array([q[0], -q[1], -q[2], -q[3]])
    else:
        return np.array([-q[0], -q[1], -q[2], q[3]])

def quat_as_matrix(q, seq='xyzw'):
    """ quaternion to rotation_matrix """
    assert seq in ('wxyz', 'xyzw')
    if seq == 'wxyz':
        w, x, y, z = q
    else:
        x, y, z, w = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z,  2 * x * y - 2 * z * w,      2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w,      1 - 2 * x * x - 2 * z * z,  2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w,      2 * y * z + 2 * x * w,      1 - 2 * x * x - 2 * y * y],
    ])

def quat_from_rotvec(v, seq='xyzw'):
    """ rotation_vector to quaternion """
    assert seq in ('wxyz', 'xyzw')
    x, y, z = v
    theta = np.sqrt(x * x + y * y + z * z)
    if theta > QUAT_EPSILON:
        w = np.cos(0.5 * theta)
        sin_half = np.sin(0.5 * theta)
        scale = sin_half / theta
        x *= scale
        y *= scale
        z *= scale
    else:
        w = 1.0
        x = 0.0
        y = 0.0
        z = 0.0
    if seq == 'wxyz':
        return np.array([w, x, y, z])
    else:
        return np.array([x, y, z, w])

def quat_as_axis_angle(q, seq='xyzw'):
    """ quaternion to axis_angle """
    assert seq in ('wxyz', 'xyzw')
    if seq == 'wxyz':
        w, x, y, z = q
    else:
        x, y, z, w = q
    theta = 2.0 * np.arccos(w)
    sin_half = np.sin(theta * 0.5)
    if sin_half < QUAT_EPSILON:
        return np.array([1, 0, 0]), 0.0
    rx = x / sin_half
    ry = y / sin_half
    rz = z / sin_half
    e = np.array([rx, ry, rz])
    return e / np.linalg.norm(e), theta

def rotate(q, v, seq='xyzw'):
    """ rotate a vector by quaternion """
    assert seq in ('wxyz', 'xyzw')
    if seq == 'wxyz':
        v0 = np.array([0, v[0], v[1], v[2]])
    else:
        v0 = np.array([v[0], v[1], v[2], 0])
    ret = hamilton_product(hamilton_product(q, v0, seq=seq), conjugate(q, seq=seq), seq=seq)
    if seq == 'wxyz':
        return np.array([ret[1], ret[2], ret[3]])
    else:
        return np.array([ret[0], ret[1], ret[2]])
