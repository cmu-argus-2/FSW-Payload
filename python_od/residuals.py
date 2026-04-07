"""
CasADi symbolic residual functions for the batch orbit determination problem.

Each function returns a CasADi expression (column vector) that is squared and
summed into the objective.  All residuals are pre-divided by their noise standard
deviation so they are dimensionless and comparably scaled.

Matches the cost functors in:
  include/navigation/pose_dynamics.hpp
  include/navigation/measurement_residuals.hpp
"""
from __future__ import annotations

from enum import Enum

import casadi as ca

from quaternion import angle_axis_to_quat, quat_conjugate, quat_inv_rotate, quat_product

# ── Physical constants ──────────────────────────────────────────────────────────
GM_EARTH = 3.9860044188e5   # km^3 / s^2


# ── Numerical integration ───────────────────────────────────────────────────────

class IntegratorType(Enum):
    """Numerical integrator Options"""
    FORWARD_EULER = "forward_euler"
    RK4           = "rk4"


class OrbitalDynamics:
    """
    Continuous-time two-body orbital dynamics with unmodelled acceleration.

    State: x = [r; v]  (6-vector, km / km·s⁻¹)
    Input: u = uma      (3-vector, km·s⁻²) — held constant over each interval

    xdot = f(r, v, uma) = [v;  a_kepler(r) + uma]
    """

    @staticmethod
    def f(r: ca.MX, v: ca.MX, uma: ca.MX) -> tuple[ca.MX, ca.MX]:
        """Return (rdot, vdot) for the continuous-time two-body + UMA dynamics."""
        rdot = v
        vdot = keplerian_accel(r) + uma
        return rdot, vdot


class NumericalIntegrator:
    """
    Numerical integrator for orbital dynamics
    """

    @staticmethod
    def forward_euler(
        r0: ca.MX, v0: ca.MX, uma: ca.MX, dt: float
    ) -> tuple[ca.MX, ca.MX]:
        """x_{k+1} = x_k + dt · f(x_k, u_k)"""
        rdot, vdot = OrbitalDynamics.f(r0, v0, uma)
        return r0 + dt * rdot, v0 + dt * vdot

    @staticmethod
    def rk4(
        r0: ca.MX, v0: ca.MX, uma: ca.MX, dt: float
    ) -> tuple[ca.MX, ca.MX]:
        """Runge-Kutta 4th order numerical integrtion"""
        k1r, k1v = OrbitalDynamics.f(r0,                          v0,                          uma)
        k2r, k2v = OrbitalDynamics.f(r0 + 0.5 * dt * k1r,         v0 + 0.5 * dt * k1v,         uma)
        k3r, k3v = OrbitalDynamics.f(r0 + 0.5 * dt * k2r,         v0 + 0.5 * dt * k2v,         uma)
        k4r, k4v = OrbitalDynamics.f(r0 +       dt * k3r,         v0 +       dt * k3v,         uma)
        r1 = r0 + (dt / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
        v1 = v0 + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        return r1, v1

    @staticmethod
    def integrate(
        r0: ca.MX, v0: ca.MX, uma: ca.MX, dt: float,
        integrator_type: IntegratorType = IntegratorType.FORWARD_EULER,
    ) -> tuple[ca.MX, ca.MX]:
        """Select integrator"""
        if integrator_type is IntegratorType.FORWARD_EULER:
            return NumericalIntegrator.forward_euler(r0, v0, uma, dt)
        if integrator_type is IntegratorType.RK4:
            return NumericalIntegrator.rk4(r0, v0, uma, dt)
        raise ValueError(f"Unknown IntegratorType: {integrator_type}")


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
    integrator_type: IntegratorType = IntegratorType.FORWARD_EULER,
) -> ca.MX:
    """
    Dynamics constraint violation (6-vector), should equal zero.

    Uncertainty in dynamics expressed through UMA time-varying parameters.
    Propagation uses a numerical integrator from the available options.

    Returns  [r1 - r1_pred; v1 - v1_pred]  which is enforced as == 0.
    """
    r1_pred, v1_pred = NumericalIntegrator.integrate(r0, v0, uma, dt, integrator_type)
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
