"""
CasADi symbolic quaternion operations.

Convention throughout: quaternion = [x, y, z, w]  (same as Eigen / Ceres).
All functions accept and return CasADi column vectors (MX or DM).
"""
import casadi as ca


def quat_product(q1: ca.MX, q2: ca.MX) -> ca.MX:
    """
    Hamilton product q1 * q2.
    Both inputs are [x, y, z, w] column vectors.
    """
    x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
    x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
    return ca.vertcat(
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    )


def quat_conjugate(q: ca.MX) -> ca.MX:
    """Conjugate / inverse of a unit quaternion: [-x, -y, -z, w]."""
    return ca.vertcat(-q[0], -q[1], -q[2], q[3])


def quat_rotate(q: ca.MX, v: ca.MX) -> ca.MX:
    """
    Rotate 3-vector v by quaternion q  (v' = R(q) * v).
    q is interpreted as the body-to-ECI rotation.
    Uses the explicit rotation matrix to avoid forming a pure quaternion product.
    """
    x, y, z, w = q[0], q[1], q[2], q[3]
    # Row-major rotation matrix R(q)
    R = ca.vertcat(
        ca.horzcat(1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)),
        ca.horzcat(2*(x*y + w*z),        1 - 2*(x*x + z*z),   2*(y*z - w*x)),
        ca.horzcat(2*(x*z - w*y),        2*(y*z + w*x),        1 - 2*(x*x + y*y)),
    )
    return ca.mtimes(R, v)


def quat_inv_rotate(q: ca.MX, v: ca.MX) -> ca.MX:
    """
    Rotate v by q^{-1}  (v' = R(q)^T * v).
    Equivalent to rotating by the conjugate quaternion.
    Used to transform a vector from ECI into the body frame.
    """
    return quat_rotate(quat_conjugate(q), v)


def angle_axis_to_quat(aa: ca.MX) -> ca.MX:
    """
    Convert an angle-axis vector  aa = angle * axis  to a unit quaternion [x,y,z,w].

    Uses a regularised norm so the expression is smooth at aa = 0:
      dq.xyz = sin(||aa||/2) / ||aa||  *  aa
      dq.w   = cos(||aa||/2)

    At aa = 0 the result is the identity quaternion [0, 0, 0, 1].
    """
    # Regularised norm avoids 0/0 at the identity rotation.
    # 1e-10 is small enough not to affect any physically meaningful rotation.
    safe_norm = ca.sqrt(ca.dot(aa, aa) + 1e-20)
    half      = safe_norm / 2.0
    sinc_half = ca.sin(half) / safe_norm   # sin(||aa||/2) / ||aa||
    return ca.vertcat(
        aa[0] * sinc_half,
        aa[1] * sinc_half,
        aa[2] * sinc_half,
        ca.cos(half),
    )
