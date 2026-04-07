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
GM_EARTH  = 3.9860044188e5   # km³/s²  — Earth gravitational parameter
R_EARTH   = 6378.137         # km      — Earth equatorial radius
J2        = 1.08262668e-3    # —       — J2 zonal harmonic coefficient
OMEGA_E   = 7.2921150e-5     # rad/s   — Earth rotation rate (z-axis, ECI)

# Exponential model parameters from U.S. Standard Atmosphere 1976
# Taken from Fundamentals of Astrodynamics and Applications, 4th Edition, by David A. Vallado
H_ELLP = [300.0, 350.0, 400.0, 450.0, 500.0, 600.0, 700.0]
NOMINAL_DENSITY = [2.418e-2, 9.518e-3, 3.725e-3, 1.585e-3, 6.967e-4, 1.454e-4]  # kg/km^3
SCALE_HEIGHT = [53.628, 53.298, 58.515, 60.828, 63.822, 71.835]  # km

# ── Numerical integration ───────────────────────────────────────────────────────

class IntegratorType(Enum):
    """Numerical integrator selection — mirrors C++ IntegratorType in pose_dynamics.hpp."""
    FORWARD_EULER = "forward_euler"
    RK4           = "rk4"


class OrbitalDynamics:
    """
    Continuous-time orbital dynamics with optional J2 and drag perturbations.

    State: x = [r; v]  (6-vector, km / km·s⁻¹)
    Input: u = uma      (3-vector, km·s⁻²) — held constant over each interval

    xdot = f(t, r, v, uma) = [v;  a_kepler(r) + a_J2(r) + a_drag(r,v) + uma]

    Parameters
    ----------
    use_j2 : bool
        Include the J2 zonal harmonic gravity perturbation.
    use_drag : bool
        Include exponential-density atmospheric drag.
    bc_inv : float
        Ballistic coefficient inverse  Cd·A/m  [km²/kg].  Only used when
        use_drag=True.
        Unit conversion: 1 m²/kg = 1×10⁻⁶ km²/kg.
        Typical 3U CubeSat (Cd=2.2, A=0.03 m², m=4 kg): bc_inv ≈ 1.65×10⁻⁸ km²/kg.
    """

    def __init__(
        self,
        use_j2:   bool          = True,
        use_drag: bool          = False,
        bc_inv:   ca.MX | float = 0.0,
    ) -> None:
        self.use_j2   = use_j2
        self.use_drag = use_drag
        self.bc_inv   = bc_inv   # may be a CasADi MX decision variable

    def f(self, t: float, r: ca.MX, v: ca.MX, uma: ca.MX) -> tuple[ca.MX, ca.MX]:
        """Return (rdot, vdot) for the configured continuous-time dynamics at time t [s]."""
        rdot = v
        vdot = keplerian_accel(r) + uma
        if self.use_j2:
            vdot = vdot + j2_accel(r)
        if self.use_drag:
            vdot = vdot + drag_accel(r, v, self.bc_inv)
        return rdot, vdot


class NumericalIntegrator:
    """
    Numerical integrator for orbital dynamics — mirrors DynamicsResidual::integrate()
    in include/navigation/pose_dynamics.hpp.

    All methods accept and return CasADi MX expressions so they compose
    transparently with the CasADi / IPOPT symbolic graph.
    """

    @staticmethod
    def forward_euler(
        dynamics: OrbitalDynamics,
        t0: float, r0: ca.MX, v0: ca.MX, uma: ca.MX, dt: float,
    ) -> tuple[ca.MX, ca.MX]:
        """x_{k+1} = x_k + dt · f(t_k, x_k, u_k)"""
        rdot, vdot = dynamics.f(t0, r0, v0, uma)
        return r0 + dt * rdot, v0 + dt * vdot

    @staticmethod
    def rk4(
        dynamics: OrbitalDynamics,
        t0: float, r0: ca.MX, v0: ca.MX, uma: ca.MX, dt: float,
    ) -> tuple[ca.MX, ca.MX]:
        """Standard RK4 — UMA is held constant over the interval."""
        k1r, k1v = dynamics.f(t0,              r0,                         v0,                         uma)
        k2r, k2v = dynamics.f(t0 + 0.5 * dt,   r0 + 0.5 * dt * k1r,        v0 + 0.5 * dt * k1v,        uma)
        k3r, k3v = dynamics.f(t0 + 0.5 * dt,   r0 + 0.5 * dt * k2r,        v0 + 0.5 * dt * k2v,        uma)
        k4r, k4v = dynamics.f(t0 + dt,          r0 +       dt * k3r,        v0 +       dt * k3v,        uma)
        r1 = r0 + (dt / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
        v1 = v0 + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        return r1, v1

    @staticmethod
    def integrate(
        dynamics: OrbitalDynamics,
        t0: float, r0: ca.MX, v0: ca.MX, uma: ca.MX, dt: float,
        integrator_type: IntegratorType = IntegratorType.FORWARD_EULER,
    ) -> tuple[ca.MX, ca.MX]:
        """Dispatch to the selected integrator."""
        if integrator_type is IntegratorType.FORWARD_EULER:
            return NumericalIntegrator.forward_euler(dynamics, t0, r0, v0, uma, dt)
        if integrator_type is IntegratorType.RK4:
            return NumericalIntegrator.rk4(dynamics, t0, r0, v0, uma, dt)
        raise ValueError(f"Unknown IntegratorType: {integrator_type}")


# ── Gravity models ──────────────────────────────────────────────────────────────
def keplerian_accel(r: ca.MX) -> ca.MX:
    """Two-body gravitational acceleration:  a = -GM / ||r||³ · r  [km/s²]."""
    r_norm = ca.sqrt(ca.dot(r, r) + 1e-10)   # safe: avoids 0^{-3} at origin
    return -GM_EARTH / (r_norm ** 3) * r


def j2_accel(r: ca.MX) -> ca.MX:
    """
    J2 zonal harmonic perturbation acceleration in ECI  [km/s²].

    Derived from the gravitational potential including the J2 term:
        U_J2 = (GM·R_E²·J2) / (2·r³) · (3·sin²φ − 1)
    where sin φ = z/r.

    The resulting ECI acceleration is (Vallado, "Fundamentals of Astrodynamics"):
        a_x = (3/2) · J2·GM·R_E²/r⁵ · x · (5(z/r)² − 1)
        a_y = (3/2) · J2·GM·R_E²/r⁵ · y · (5(z/r)² − 1)
        a_z = (3/2) · J2·GM·R_E²/r⁵ · z · (5(z/r)² − 3)
    """
    r_sq   = ca.dot(r, r) + 1e-10             # ||r||²  (safe)
    r_norm = ca.sqrt(r_sq)                    # ||r||
    r5     = r_sq * r_sq * r_norm             # ||r||⁵

    factor      = 1.5 * J2 * GM_EARTH * R_EARTH ** 2 / r5
    z_sq_over_r_sq = r[2] * r[2] / r_sq

    ax = factor * r[0] * (5.0 * z_sq_over_r_sq - 1.0)
    ay = factor * r[1] * (5.0 * z_sq_over_r_sq - 1.0)
    az = factor * r[2] * (5.0 * z_sq_over_r_sq - 3.0)
    return ca.vertcat(ax, ay, az)


# ── Atmospheric drag ────────────────────────────────────────────────────────────

def _atmospheric_density(r: ca.MX) -> ca.MX:
    """
    Piecewise exponential atmospheric density at satellite position r [km]  →  [kg/km³].

    Layer i covers h ∈ [H_ELLP[i], H_ELLP[i+1]):
        ρ_i(h) = NOMINAL_DENSITY[i] · exp(-(h − H_ELLP[i]) / SCALE_HEIGHT[i])

    Uses nested ca.if_else so the expression remains a valid CasADi symbolic.
    Outside the table range the nearest layer is extrapolated (clamped index).

    Reference: Vallado, "Fundamentals of Astrodynamics and Applications", Table 8-4.
    """
    r_norm = ca.sqrt(ca.dot(r, r) + 1e-10)   # ||r||  [km]
    h_ellp = r_norm - R_EARTH                 # altitude above ellipsoid [km]

    # Build the piecewise expression from the top layer downward.
    # Start with the highest layer (extrapolates above H_ELLP[-1]).
    density = NOMINAL_DENSITY[-1] * ca.exp(-(h_ellp - H_ELLP[-2]) / SCALE_HEIGHT[-1])

    # Walk down: at each boundary, override with the lower layer if h is below it.
    for i in range(len(NOMINAL_DENSITY) - 2, -1, -1):
        layer_i = NOMINAL_DENSITY[i] * ca.exp(-(h_ellp - H_ELLP[i]) / SCALE_HEIGHT[i])
        density = ca.if_else(h_ellp < H_ELLP[i + 1], layer_i, density)

    return density


def drag_accel(r: ca.MX, v: ca.MX, bc_inv: ca.MX | float) -> ca.MX:
    """
    Atmospheric drag acceleration in ECI  [km/s²].

        a_drag = −½ · bc_inv · ρ(h) · ||v_rel|| · v_rel

    where v_rel is the satellite velocity relative to the co-rotating atmosphere:

        v_atm   = ω_E × r  =  [−ω_E·y,  ω_E·x,  0]   (atmosphere co-rotates with Earth)
        v_rel   = v_ECI − v_atm

    Parameters
    ----------
    r      : ECI position  [km]
    v      : ECI velocity  [km/s]
    bc_inv : Cd·A/m  [km²/kg]
             Unit note: 1 m²/kg = 1×10⁻⁶ km²/kg
    """
    # Atmosphere velocity at satellite position (ECI): ω_E × r
    # ω_E = [0, 0, OMEGA_E]  →  ω_E × r = [−OMEGA_E·y, OMEGA_E·x, 0]
    v_atm = ca.vertcat(-OMEGA_E * r[1], OMEGA_E * r[0], ca.MX(0.0))
    v_rel = v - v_atm                                   # [km/s]

    rho         = _atmospheric_density(r)               # [kg/km³]
    v_rel_norm  = ca.sqrt(ca.dot(v_rel, v_rel) + 1e-20) # [km/s]  (safe)

    return -0.5 * (1.0 + bc_inv) * rho * v_rel_norm * v_rel


def drag_prior_residual(
    bc_inv:          ca.MX,
    bc_inv_nominal:  float,
    bc_inv_std:      float,
) -> ca.MX:
    """
    Gaussian prior on the drag ballistic coefficient inverse (scalar).

    Penalises deviation from bc_inv_nominal with standard deviation bc_inv_std,
    both in km²/kg.  Analogous to uma_prior_residual.
    """
    return (bc_inv - bc_inv_nominal) / bc_inv_std


# ── Orbital dynamics constraint ─────────────────────────────────────────────────

def linear_dynamics_constraint(
    r0: ca.MX, v0: ca.MX,
    r1: ca.MX, v1: ca.MX,
    uma: ca.MX,
    t0: float,
    dt: float,
    integrator_type: IntegratorType = IntegratorType.FORWARD_EULER,
    dynamics: OrbitalDynamics | None = None,
) -> ca.MX:
    """
    Dynamics constraint violation (6-vector), should equal zero.

    Propagates the state [r0, v0] forward by dt using the chosen integrator and
    physics model, then returns [r1 − r1_pred; v1 − v1_pred], which IPOPT
    enforces as == 0.

    Parameters
    ----------
    t0 : float
        Absolute time at the start of the interval [s].
    dt : float
        Interval duration [s].
    integrator_type : IntegratorType
        FORWARD_EULER or RK4.
    dynamics : OrbitalDynamics | None
        Physics configuration.  Defaults to two-body Keplerian only.
    """
    if dynamics is None:
        dynamics = OrbitalDynamics()
    r1_pred, v1_pred = NumericalIntegrator.integrate(
        dynamics, t0, r0, v0, uma, dt, integrator_type
    )
    return ca.vertcat(r1 - r1_pred, v1 - v1_pred)


def uma_prior_residual(uma: ca.MX, uma_std: float) -> ca.MX:
    """
    Gaussian prior on unmodelled acceleration (3-vector).
    Penalises deviations from zero with standard deviation uma_std [km/s²].
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
