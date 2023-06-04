""" Quaternion Algebra """
import numpy as np

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
