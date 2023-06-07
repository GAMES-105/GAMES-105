""" Jacobian-based IK Solver 
    References:
    http://www.andreasaristidou.com/publications/papers/IK_survey.pdf

Find Jacobian matrix:
Denote forward kinematics as the function of 
F(θ) = {p_0, p_1, p_2, ..., p_m}
    which computes m rotations, m links, m+1 points

    Joint: Q_i as global rotation, R_i as local rotation (i=0,1,2,...,m-1)
    Link:  l_i as length from point i to i+1             (i=0,1,2,...,m-1)
    Positions: p_i (i=0,1,2,...,m)

    Q_i = Q_{i-1} * R_{i}
    p_i = p_{i-1} + Q_{i-1} * l_{i-1}

Suppose there are `m` rotations and `n` target points, i.e.
F(θ): R^3m -> R^3n
    F(θ) = [ F(θ)_0, F(θ)_1, ..., F(θ)_j, ..., F(θ)_{n-1} ]

    thus the dimension of Jacobian matrix shall be of `n*m` (`3n*3m`)
    J(θ) = ∂F/∂θ
         = [ ∂F/∂θ_0, ∂F/∂θ_1, ..., ∂F/∂θ_i, ..., ∂F/∂θ_{m-1} ]
         = [ ...,                             ]
             ...
           [ ..., ..., ∂F(θ)_j/∂θ_i, ..., ... ]
             ...
           [ ...,                             ]

Consider the target j as x_j, the function item of F(θ)_j, and the corresponding ∂F/∂θ should be:

    F(θ)_j = p_j 
           = p_0 + 
             R_0 * l_0 + 
             R_0 * R_1 * l_1 + 
             R_0 * R_1 * R_2 * l_2 + 
             ... + 
             R_0 * R_1 * ... * R_{i-1} * l_{i-1} + 
             R_0 * R_1 * ... * R_{i-1} * R_i * l_i + 
             ... + 
             R_0 * R_1 * ... * R_{i-1} * R_i * R_{i+1} * ... * R_{j-1} * l_{j-1}

    notice that:
        1. everything before R_i has nothing with R_i, which means it could be ignored 
           during computing the partial derivative
        2. product_{k=0,...,i-1}{R_k} = Q_{i-1}, i.e. the global rotation of joint i's parent
        3. let: rho_{i,j} = l_i + R_{i+1} * l_{i+1} + ... + R_{i+1} * R_{i+2} * ... * R_{j-1} * l_{j-1}
    then F(θ)_j can be expressed as

    F(θ)_j = ____ + 
             Q_{i-1} * R_i * rho_{i,j}
    ∂F_j/∂θ_i
           = lim_{δθ_i -> 0} { ( Q_{i-1} * exp(δθ_i^) * R_i * rho_{i,j} - Q_{i-1} * R_i * rho_{i,j} ) / δθ_i }
           = lim_{δθ_i -> 0} { ( Q_{i-1} * (I + δθ_i^) * R_i * rho_{i,j} - Q_{i-1} * R_i * rho_{i,j} ) / δθ_i }
           = lim_{δθ_i -> 0} { Q_{i-1} * ( δθ_i^ * R_i * rho_{i,j} / δθ_i ) }
           = lim_{δθ_i -> 0} { - Q_{i-1} * (R_i * rho_{i,j})^ * δθ_i / δθ_i }
           = - Q_{i-1} * (R_i * rho_{i,j})^

    notice that:
        1. rho_{i,j} is the local position of joint j in joint i
            since:
                p_j = p_i + Q_i * l_i + Q_i * R_{i+1} * l_{i+1} + ... + Q_i * R_{i+1} * ... * R_{j-1} * l_{j-1}
                    = p_i + Q_i * rho_{i,j}
            i.e. 
                rho_{i,j} = (Q_i)^-1 * (p_j - p_i)
            which means rho_{i,j} can be calculated easily from global rotations and positions ...
            =>
                rho_{i,j} = Q_i.T * (p_j - p_i)
            => Q_i.T ~ the transpose (indeed, the inverse) of j's global rotation
            => (p_j - p_i) ~ the vector from i to j, in global position
        2. Q_{i-1} is the global rotation, while R_i is the local rotation
        3. ^ as skew matrix

For convenience, let n = 1, then that Jacobian shall be of `3*3m`
Input:
    global_positions: (m+1, 3) ndarray
    global_rotations: (m, 4)   ndarray
    target_points:    (3,)     ndarray

"""

import numpy as np

from quaternion import hamilton_product, conjugate, to_axis_angle, rotate
from scipy.spatial.transform import Rotation as R

def skew_matrix(v):
    """ Convert 3d vector to skew matrix """
    x, y, z = v
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]])

def jacobian(gp, gr, t):
    """ Single target jacobian """
    assert gp.shape[-1] == 3
    assert gr.shape[-1] == 4
    assert t.shape == (3,)
    assert gp.shape[0] == gr.shape[0] + 1
    m = gr.shape[0]
    J = np.empty((t.shape[0], 3 * m))
    for i in range(m):
        Q_i_1 = gr[i - 1] if i > 0 else np.array([0, 0, 0, 1]) # (seq=xyzw)identity
        Q_i = gr[i]
        R_i = hamilton_product(conjugate(Q_i_1), Q_i)
        rho_i_j = rotate(conjugate(Q_i), gp[-1] - gp[i])
        # J[:, i] = R.from_matrix( R.from_quat(Q_i_1).as_matrix() @ skew_matrix(rotate(R_i, rho_i_j)) ).as_rotvec()
        J[:, 3 * i: 3 * (i + 1)] = R.from_quat(Q_i_1).as_matrix() @ skew_matrix(rotate(R_i, rho_i_j))
    return J

def jacobian_cross(gp, gr, t):
    """ Single target jacobian, by axis cross """
    # DEBUG:
    ### i'm so confused of its dimensions
    ### still something abnormal ...
    assert gp.shape[-1] == 3
    assert gr.shape[-1] == 4
    assert t.shape == (3,)
    assert gp.shape[0] == gr.shape[0] + 1
    m = gr.shape[0]
    J = np.empty((t.shape[0], m))
    for i in range(m):
        R_i = gr[i] if i == 0 else hamilton_product(conjugate(gr[i - 1]), gr[i])
        v_i, _ = to_axis_angle(R_i)
        # aa_i = R.from_quat(R_i).as_rotvec()
        # v_i = aa_i / (np.linalg.norm(aa_i) + 1E-8)
        J[:, i] = np.cross(v_i, gp[-1] - gp[i])
    return J


def jacobian_transpose_ik(gp, gr, offset, t):
    """ delta_θ = alpha * J_T * e """
    n_iter = 2000
    error = np.linalg.norm(t - gp[-1])
    print(f"[Jacobian Transpose IK] before: error = {error}")
    for i in range(n_iter):
        J = jacobian(gp, gr, t)
        e = t - gp[-1]
        j_jt_e = J @ J.T @ e
        alpha = e.dot(j_jt_e) / (j_jt_e.dot(j_jt_e) + 1E-8)
        delta = alpha * J.T @ e

        # print(delta.shape)
        # print(delta)

        lr = [q if j == 0 else hamilton_product(conjugate(gr[j - 1]), q) for j, q in enumerate(gr)]
        for j, q in enumerate(lr):
            # # DEBUG: test if q not modified
            # lr[j] = q

            # DEBUG; how to update local rotations?
            ###      use rotvec directly adding, or use rotate
            # lr[j] = R.from_rotvec(R.from_quat(q).as_rotvec() - delta[3 * j: 3 * (j + 1)]).as_quat()
            lr[j] = ( R.from_quat(q) * R.from_rotvec(delta[3 * j: 3 * (j + 1)]) ).as_quat()

            gr[j] = lr[j] if j == 0 else hamilton_product(gr[j - 1], lr[j])
            gp[j + 1] = gp[j] + rotate(gr[j], offset[j])

        error = np.linalg.norm(t - gp[-1])
        print(f"[Iter:{i:02d}] in progress: alpha = {alpha:.08f}, error = {error:.08f}")


if __name__ == '__main__':

    # np.random.seed(0xac1997)

    n = 100
    lRs = np.random.rand(n, 4) + 1E-4 # local rotations
    lRs /= np.linalg.norm(lRs, axis=-1, keepdims=True) # unit quaternion
    bone = np.random.rand(n, 3) + 0.05 # local bone vectors
    gRs = []
    gPs = [np.array([0, 0, 0]), ]
    for i in range(n):
        gR = lRs[i] if i == 0 else hamilton_product(gRs[-1], lRs[i])
        gP = gPs[-1] + rotate(gR, bone[i])
        gRs.append(gR)
        gPs.append(gP)

    gRs = np.array(gRs)
    gPs = np.array(gPs)
    print(gRs.shape)
    print(gPs.shape)
    target = gPs[-1] + 0.5 # add more in R^3

    J1 = jacobian(gp=gPs, gr=gRs, t=target)
    J2 = jacobian_cross(gp=gPs, gr=gRs, t=target)
    print(J1.shape)
    print(J2.shape)
    # print(J1)
    # print(J2)

    # method-1
    jacobian_transpose_ik(gp=gPs, gr=gRs, offset=bone, t=target)


