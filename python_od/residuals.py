"""
CasADi symbolic residual functions for the batch orbit determination problem.

Each function returns a CasADi expression (column vector) that is squared and
summed into the objective.  All residuals are pre-divided by their noise standard
deviation so they are dimensionless and comparably scaled.

Matches the cost functors in:
  include/navigation/pose_dynamics.hpp
  include/navigation/measurement_residuals.hpp
"""
import casadi as ca

from quaternion import angle_axis_to_quat, quat_conjugate, quat_inv_rotate, quat_product

# ── Physical constants ──────────────────────────────────────────────────────────
GM_EARTH = 3.9860044188e5   # km^3 / s^2


# ── Linear (orbital) dynamics ───────────────────────────────────────────────────

def keplerian_accel(r: ca.MX) -> ca.MX:
    """Two-body gravitational acceleration:  a = -GM / ||r||^3 * r  [km/s^2]."""
    r_norm = ca.sqrt(ca.dot(r, r) + 1e-10)   # safe: avoids 0^{-3} at origin
    return -GM_EARTH / (r_norm ** 3) * r


def linear_dynamics_constraint(
    r0: ca.MX, v0: ca.MX,
    r1: ca.MX, v1: ca.MX,
    uma: ca.MX,
    dt: float,
) -> ca.MX:
    """
    Forward-Euler dynamics constraint violation (6-vector), should equal zero.

    The unmodelled acceleration (UMA) corrects the two-body model:
        r1_pred = r0 + v0 * dt + 0.5 * uma * dt²
        v1_pred = v0 + (a_kepler(r0) + uma) * dt

    Returns  [r1 - r1_pred; v1 - v1_pred]  which is enforced as == 0.
    """
    a0      = keplerian_accel(r0)
    r1_pred = r0 + v0 * dt + 0.5 * uma * dt ** 2
    v1_pred = v0 + (a0 + uma) * dt
    return ca.vertcat(r1 - r1_pred, v1 - v1_pred)


def uma_prior_residual(uma: ca.MX, uma_std: float) -> ca.MX:
    """
    Gaussian prior on unmodelled acceleration (3-vector).
    Penalises deviations from zero with standard deviation uma_std [km/s^2].
    """
    return uma / uma_std


# ── Angular (attitude) dynamics — fixed bias ────────────────────────────────────

def angular_dynamics_residual_fix_bias(
    q0: ca.MX,
    q1: ca.MX,
    gyro_w: ca.MX,   # measured angular velocity [rad/s], can be DM constant
    bias:   ca.MX,   # fixed gyro bias [rad/s], symbolic decision variable
    dt: float,
    quat_std: float,
) -> ca.MX:
    """
    Fixed-bias quaternion-dynamics residual (3-vector).

    Predicted next quaternion:
        omega   = gyro_w - bias               (unbiased angular velocity)
        dq      = angle_axis_to_quat(omega * dt)
        q1_pred = q0 * dq

    Error quaternion:
        q_err   = q1_pred^{*} * q1

    Residual: imaginary part of q_err, normalised by quat_std.
    For small attitude errors this equals the angle-axis error (Ceres uses the
    exact QuaternionToAngleAxis conversion; the difference is negligible when
    the optimiser is well-initialised).
    """
    omega  = gyro_w - bias
    dq     = angle_axis_to_quat(omega * dt)
    q_pred = quat_product(q0, dq)
    q_err  = quat_product(quat_conjugate(q_pred), q1)
    # q_err[:3] is the imaginary (vector) part — proportional to sin(angle/2)*axis
    return q_err[:3] / quat_std


# ── Landmark bearing measurement ────────────────────────────────────────────────

def landmark_residual(
    r:            ca.MX,   # satellite ECI position [km]
    q:            ca.MX,   # body-to-ECI quaternion [x,y,z,w]
    lmk_pos:      ca.MX,   # landmark ECI position [km], DM constant
    bearing_meas: ca.MX,   # measured bearing unit vector in body frame, DM constant
    lmk_std:      float,
) -> ca.MX:
    """
    Landmark bearing residual (3-vector).

    Predicted bearing in ECI:
        bearing_eci = (lmk_pos - r) / ||lmk_pos - r||

    Rotate into body frame (q transforms body -> ECI, so inverse rotates ECI -> body):
        bearing_body_pred = R(q)^T * bearing_eci

    Residual = (predicted_body - measured_body) / lmk_std
    """
    diff         = lmk_pos - r
    dist         = ca.sqrt(ca.dot(diff, diff) + 1e-6)   # safe norm [km]
    bearing_eci  = diff / dist
    bearing_body = quat_inv_rotate(q, bearing_eci)
    return (bearing_body - bearing_meas) / lmk_std
