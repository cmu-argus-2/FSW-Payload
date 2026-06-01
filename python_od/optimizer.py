"""
CasADi/IPOPT batch orbit determination optimizer — fixed gyro bias, constrained dynamics.

Decision variables:
  r[i]  — ECI position        (3,)  [km]        per state
  v[i]  — ECI velocity        (3,)  [km/s]      per state
  q[i]  — body-to-ECI quat    (4,)  [x,y,z,w]  per state  (||q||=1 constraint)
  a[i]  — unmodelled accel    (3,)  [km/s²]     per interval
  b     — fixed gyro bias     (3,)  [rad/s]     global

Equality constraints (IPOPT enforces these to machine precision):
  linear dynamics:  propagate([r_i; v_i]) via Forward Euler or RK4 using
                    f(r,v) = [v; a_kepler(r) + a_J2(r) + a_drag(r,v) + a_i]
                    (J2 and drag are optional; a_i is the per-interval UMA)
  quaternion norm:  ||q_i||² = 1

Soft cost (quadratic):
  • UMA prior            ‖a_i / σ_a‖²         encourages small unmodelled forces
  • Angular dynamics     ‖q_err_i / σ_q‖²     gyro integration residual
  • Landmark bearing     ‖Δbearing / σ_lmk‖²  vision measurement residual

Enforcing dynamics as equality constraints lets σ_a be set to the true expected
magnitude of unmodelled forces (e.g. 1e-4 km/s²) without ill-conditioning —
IPOPT's interior-point solver handles the constraint structure directly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import casadi as ca
import numpy as np
import scipy.linalg

from data_loader import get_state_timestamps
from residuals import (
    IntegratorType,
    OrbitalDynamics,
    angular_dynamics_residual_fix_bias,
    drag_prior_residual,
    landmark_residual,
    linear_dynamics_constraint,
    pseudo_huber_cost,
    uma_prior_residual,
)

# ── Noise / prior parameters ───────────────────────────────────────────────────
UMA_STD_DEV     = 1e-5        # km/s²  — expected magnitude of unmodelled forces
GYRO_WN_STD_DEV = 0.0008726   # rad/s  — gyro white noise std dev
LANDMARK_STD    = 0.009       # rad    — landmark bearing noise

R_ORBIT_KM = 6371.0 + 600.0  # default orbital altitude for position initialisation


# ── Trajectory initializer ───────────────────────────────────────────────────────

def _wahba_svd(B: np.ndarray) -> np.ndarray:
    """
    Wahba's problem via SVD: find rotation R minimising sum‖d_eci − R·b_body‖².
    B = Σ d_eci · b_body^T.  Returns identity quaternion [x,y,z,w] on degenerate input.
    """
    if np.linalg.norm(B) < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    U, _, Vt = np.linalg.svd(B)
    diag = np.array([1.0, 1.0, np.linalg.det(U @ Vt)])
    R = U @ np.diag(diag) @ Vt
    # Convert rotation matrix to quaternion [x, y, z, w]
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """SLERP between two unit quaternions [x,y,z,w]."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        return (q0 + alpha * (q1 - q0)) / np.linalg.norm(q0 + alpha * (q1 - q0))
    theta0 = np.arccos(dot)
    theta  = theta0 * alpha
    sin0   = np.sin(theta0)
    return np.sin(theta0 - theta) / sin0 * q0 + np.sin(theta) / sin0 * q1


class TrajectoryInitializer:
    """
    Mirrors C++ TrajectoryInitializer in src/navigation/trajectory_initializer.cpp.

    Per-group pose keyframes:
      position  = mean landmark ECI position renormalised to R_ORBIT_KM
      attitude  = Wahba SVD on (satellite-to-landmark ECI direction, body bearing) pairs

    All states:
      position  = linear interpolation/extrapolation between keyframes
      attitude  = SLERP between keyframes
      velocity  = central finite differences of position

    Gyro bias (fixed):
      mean over all states of (ω_meas − ω_att),
      where ω_att ≈ 2·(q₀⁻¹⊗q₁).vec / dt
    """

    def __init__(
        self,
        state_timestamps:       list[float],
        landmark_measurements:  np.ndarray,   # (N, 7): [t, bx, by, bz, ex, ey, ez]
        landmark_group_starts:  np.ndarray,   # (N,) bool
        gyro_measurements:      np.ndarray,   # (M, 4): [t, wx, wy, wz] rad/s
        estimate_bias:          bool = True,
    ) -> None:
        ts = np.array(state_timestamps)
        M  = len(ts)
        N_lmk = len(landmark_measurements)

        # ── Build pose keyframes (one per landmark group) ─────────────────────
        keyframe_t:   list[float]      = []
        keyframe_pos: list[np.ndarray] = []
        keyframe_quat: list[np.ndarray] = []

        i = 0
        while i < N_lmk:
            if not landmark_group_starts[i]:
                i += 1
                continue

            t_group = landmark_measurements[i, 0]
            lm_pos_list:  list[np.ndarray] = []
            bearing_list: list[np.ndarray] = []

            # collect all rows belonging to this group
            while i < N_lmk and (len(lm_pos_list) == 0 or not landmark_group_starts[i]):
                lm_pos_list.append(landmark_measurements[i, 4:7].copy())
                bearing_list.append(landmark_measurements[i, 1:4].copy())
                i += 1

            if not lm_pos_list:
                continue

            pos_sum = sum(lm_pos_list)
            norm    = np.linalg.norm(pos_sum)
            if norm < 1e-6:
                continue
            r_group = pos_sum / norm * R_ORBIT_KM

            B = np.zeros((3, 3))
            for lm_pos, bearing in zip(lm_pos_list, bearing_list):
                diff = lm_pos - r_group
                d    = np.linalg.norm(diff)
                if d < 1e-6:
                    continue
                B += np.outer(diff / d, bearing)

            keyframe_t.append(t_group)
            keyframe_pos.append(r_group)
            keyframe_quat.append(_wahba_svd(B))

        # ── Interpolate to all state timestamps ───────────────────────────────
        positions  = np.zeros((M, 3))
        quaternions = np.zeros((M, 4))
        quaternions[:, 3] = 1.0   # identity [x,y,z,w]

        if not keyframe_t:
            positions[:, 2] = R_ORBIT_KM
        else:
            kf_t = np.array(keyframe_t)
            kf_p = np.array(keyframe_pos)    # (K, 3)
            kf_q = np.array(keyframe_quat)   # (K, 4)
            K    = len(kf_t)

            for j, t in enumerate(ts):
                if K == 1:
                    positions[j]  = kf_p[0]
                    quaternions[j] = kf_q[0]
                elif t <= kf_t[0]:
                    # Extrapolate before first keyframe
                    dt = kf_t[1] - kf_t[0]
                    if abs(dt) < 1e-9:
                        positions[j] = kf_p[0]
                    else:
                        positions[j] = kf_p[0] + (kf_p[1] - kf_p[0]) / dt * (t - kf_t[0])
                    quaternions[j] = kf_q[0]
                elif t >= kf_t[-1]:
                    dt = kf_t[-1] - kf_t[-2]
                    if abs(dt) < 1e-9:
                        positions[j] = kf_p[-1]
                    else:
                        positions[j] = kf_p[-1] + (kf_p[-1] - kf_p[-2]) / dt * (t - kf_t[-1])
                    quaternions[j] = kf_q[-1]
                else:
                    idx = int(np.searchsorted(kf_t, t, side="right")) - 1
                    idx = max(0, min(idx, K - 2))
                    alpha = (t - kf_t[idx]) / (kf_t[idx + 1] - kf_t[idx])
                    positions[j] = (1.0 - alpha) * kf_p[idx] + alpha * kf_p[idx + 1]
                    quaternions[j] = _slerp(kf_q[idx], kf_q[idx + 1], alpha)

        # ── Renormalize positions to orbital altitude ─────────────────────────
        # Linear interpolation between keyframes in very different ECI directions
        # (e.g. from bad landmark groups) can produce positions inside Earth.
        # Renormalising to R_ORBIT_KM keeps the initial guess physically valid.
        for j in range(M):
            norm = np.linalg.norm(positions[j])
            if norm > 1e-6:
                positions[j] = positions[j] / norm * R_ORBIT_KM

        # ── Velocity: central finite differences ──────────────────────────────
        velocities = np.zeros((M, 3))
        for j in range(M):
            i0 = max(0, j - 1)
            i1 = min(M - 1, j + 1)
            if i0 == i1:
                continue
            dt = ts[i1] - ts[i0]
            velocities[j] = (positions[i1] - positions[i0]) / dt

        # ── Gyro bias: mean(ω_meas − ω_att) ──────────────────────────────────
        gyro_bias = np.zeros(3)
        if estimate_bias and len(gyro_measurements) > 0:
            Mg = len(gyro_measurements)
            Nb = min(M, Mg)
            bias_sum = np.zeros(3)
            count    = 0
            for j in range(Nb):
                i0 = j if j < M - 1 else j - 1
                i1 = i0 + 1
                dt = ts[i1] - ts[i0]
                if abs(dt) < 1e-9:
                    continue
                # q0^{-1} ⊗ q1 — Hamilton product with q0 conjugated
                q0  = quaternions[i0]  # [x,y,z,w]
                q1  = quaternions[i1]
                # conjugate of q0: [-x,-y,-z,w]
                q0c = np.array([-q0[0], -q0[1], -q0[2], q0[3]])
                # Hamilton product q0c ⊗ q1
                x0, y0, z0, w0 = q0c
                x1, y1, z1, w1 = q1
                dq_vec = np.array([
                    w0*x1 + x0*w1 + y0*z1 - z0*y1,
                    w0*y1 - x0*z1 + y0*w1 + z0*x1,
                    w0*z1 + x0*y1 - y0*x1 + z0*w1,
                ])
                omega_att = 2.0 * dq_vec / dt
                omega_meas = gyro_measurements[j, 1:4]
                bias_sum += omega_meas - omega_att
                count    += 1
            if count > 0:
                gyro_bias = bias_sum / count

        self.positions   = positions    # (M, 3)
        self.velocities  = velocities   # (M, 3)
        self.quaternions = quaternions  # (M, 4)  [x,y,z,w]
        self.gyro_bias   = gyro_bias    # (3,)


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _build_landmark_groups(
    landmark_group_starts: np.ndarray,
    lmk_group_indices: list[int],
) -> list[tuple[int, list[int]]]:
    """Return a list of (state_idx, [measurement_row_indices]) per landmark group."""
    groups: list[tuple[int, list[int]]] = []
    group_k = 0
    for row in range(len(landmark_group_starts)):
        if landmark_group_starts[row]:
            groups.append((lmk_group_indices[group_k], []))
            group_k += 1
        groups[-1][1].append(row)
    return groups


def _filter_landmark_measurements(
    landmark_measurements:  np.ndarray,
    landmark_group_starts:  np.ndarray,
    landmark_uncertainties: np.ndarray | None,
    keep_mask:              np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Return filtered landmark arrays keeping only rows where keep_mask is True.

    Reconstructs landmark_group_starts so the first surviving row of each
    original group is marked as a group start.  Groups that lose all rows are
    removed entirely.
    """
    group_ids = np.cumsum(landmark_group_starts).astype(int) - 1  # 0-based group index per row

    filtered_meas      = landmark_measurements[keep_mask]
    filtered_group_ids = group_ids[keep_mask]

    if len(filtered_meas) == 0:
        empty_unc = np.array([], dtype=np.float64) if landmark_uncertainties is not None else None
        return filtered_meas, np.array([], dtype=bool), empty_unc

    new_group_starts     = np.zeros(len(filtered_group_ids), dtype=bool)
    new_group_starts[0]  = True
    new_group_starts[1:] = filtered_group_ids[1:] != filtered_group_ids[:-1]

    filtered_unc = landmark_uncertainties[keep_mask] if landmark_uncertainties is not None else None
    return filtered_meas, new_group_starts, filtered_unc


def _quat_tangent_basis(q: np.ndarray) -> np.ndarray:
    """
    4×3 orthonormal basis for the tangent space of unit quaternion q.
    The tangent space is the 3D subspace of R^4 orthogonal to q.
    """
    q = q / np.linalg.norm(q)
    A = np.empty((4, 4))
    A[:, 0] = q
    A[:, 1:] = np.eye(4)[:, :3]
    Q, _ = np.linalg.qr(A)
    return Q[:, 1:]   # 4×3


def _build_tangent_projection(
    q_vals: np.ndarray, N: int, N_uma: int, use_cd: bool = False
) -> np.ndarray:
    """
    Build the (n_full) × (n_reduced) tangent-space projection matrix T.

    Full parameter vector  x = [vec(r); vec(v); vec(q); vec(a); b; (cd)]
    sizes:                      3N       3N       4N      3·N_uma  3   (1)
    n_full = 10N + 3·N_uma + 3 (+ 1 if use_cd)

    Reduced (tangent) vector x_t = [vec(r); vec(v); vec(q_t); vec(a); b; (cd)]
    sizes:                           3N       3N       3N       3·N_uma  3   (1)
    n_reduced = 9N + 3·N_uma + 3 (+ 1 if use_cd)

    Only the quaternion blocks need non-trivial projection (4→3 per state).
    All other blocks are identity.
    """
    n_cd      = 1 if use_cd else 0
    n_full    = 10 * N + 3 * N_uma + 3 + n_cd
    n_reduced =  9 * N + 3 * N_uma + 3 + n_cd
    T = np.zeros((n_full, n_reduced))

    # r block
    T[:3*N, :3*N] = np.eye(3 * N)

    # v block
    T[3*N:6*N, 3*N:6*N] = np.eye(3 * N)

    # q tangent blocks
    for i in range(N):
        B = _quat_tangent_basis(q_vals[i])   # 4×3
        T[6*N + 4*i : 6*N + 4*i + 4,
          6*N + 3*i : 6*N + 3*i + 3] = B

    # a (UMA) block — no manifold constraint, identity pass-through
    a_row = 10 * N
    a_col =  9 * N
    T[a_row : a_row + 3*N_uma, a_col : a_col + 3*N_uma] = np.eye(3 * N_uma)

    # b (bias) block
    T[a_row + 3*N_uma : a_row + 3*N_uma + 3,
      a_col + 3*N_uma : a_col + 3*N_uma + 3] = np.eye(3)

    # cd block (scalar, identity) — only present when estimating drag
    if use_cd:
        T[-1, -1] = 1.0

    return T


def _compute_covariance(
    cost_residuals_sym: list[ca.MX],
    dyn_constraint_syms: list[ca.MX],
    sol: ca.OptiSol,
    r: ca.MX,
    v: ca.MX,
    q: ca.MX,
    a: ca.MX,
    b: ca.MX,
    N: int,
    cd: ca.MX | None = None,
) -> dict[str, np.ndarray]:
    """
    Approximate covariance from the augmented Jacobian at the solution.

    For a constrained least-squares problem the covariance comes from:
        Σ ≈ (J_aug^T J_aug)^{-1}
    where J_aug stacks:
      • The Jacobian of all soft cost residuals (angular, landmark, UMA prior,
        and drag prior when cd is estimated)
      • The Jacobian of the dynamics equality constraints, scaled by a tight
        normalisation factor — this captures the constraint coupling between
        states and UMA without needing to form the full KKT matrix.

    Quaternion blocks are projected onto their 3D tangent space.

    Returns
    -------
    dict with:
        pos_var    : (N, 3)       position variance        [km²]
        vel_var    : (N, 3)       velocity variance        [(km/s)²]
        rot_var    : (N, 3)       attitude tangent variance [rad²]
        uma_var    : (N_uma, 3)   UMA variance             [(km/s²)²]
        bias_var   : (3,)         gyro bias variance       [(rad/s)²]
        cd_var     : float        drag cd variance         [-]  (only if cd given)
    """
    print("[covariance] Building symbolic Jacobian …")

    N_uma      = N - 1
    use_cd = cd is not None

    # Full symbolic parameter vector (column-major vectorisation)
    x_sym = ca.vertcat(ca.vec(r), ca.vec(v), ca.vec(q), ca.vec(a), b)
    if use_cd:
        x_sym = ca.vertcat(x_sym, cd)

    DYN_NORM = 1e-6   # km / (km/s) representative scale — tunable
    dyn_sym  = ca.vertcat(*[c / DYN_NORM for c in dyn_constraint_syms])

    r_sym = ca.vertcat(*cost_residuals_sym, dyn_sym)

    J_sym = ca.jacobian(r_sym, x_sym)
    J_fn  = ca.Function("J", [x_sym], [J_sym])

    # Evaluate at solution
    r_val = sol.value(r)              # (3, N)
    v_val = sol.value(v)              # (3, N)
    q_val = sol.value(q)              # (4, N)
    a_val = np.asarray(sol.value(a)) if N_uma > 1 else np.asarray(sol.value(a)).reshape(3, 1)
    b_val = sol.value(b)              # (3,)

    x_val = np.concatenate([
        r_val.ravel(order="F"),
        v_val.ravel(order="F"),
        q_val.ravel(order="F"),
        a_val.ravel(order="F"),
        np.asarray(b_val).ravel(),
    ])
    if use_cd:
        x_val = np.concatenate([x_val, [float(sol.value(cd))]])

    print("[covariance] Evaluating Jacobian at solution …")
    J_val = np.array(J_fn(x_val))

    print("[covariance] Projecting to tangent space …")
    T = _build_tangent_projection(q_val.T, N, N_uma, use_cd=use_cd)
    J_reduced = J_val @ T

    # Diagonal of (J^T J)^{-1} via SVD — avoids forming J^T J explicitly
    print("[covariance] Computing SVD …")
    _, S, Vt = scipy.linalg.svd(J_reduced, full_matrices=False)

    tol      = S[0] * max(J_reduced.shape) * np.finfo(float).eps * 1e3
    S_safe   = np.where(S > tol, S, np.inf)
    cov_diag = np.sum((Vt.T / S_safe) ** 2, axis=1)

    pos_var  = cov_diag[             :  3*N         ].reshape(N,     3, order="F")
    vel_var  = cov_diag[ 3*N         :  6*N         ].reshape(N,     3, order="F")
    rot_var  = cov_diag[ 6*N         :  9*N         ].reshape(N,     3, order="F")
    uma_var  = cov_diag[ 9*N         :  9*N+3*N_uma ].reshape(N_uma, 3, order="F")
    bias_var = cov_diag[ 9*N+3*N_uma :  9*N+3*N_uma+3 ]

    result = {
        "pos_var":  pos_var,
        "vel_var":  vel_var,
        "rot_var":  rot_var,
        "uma_var":  uma_var,
        "bias_var": bias_var,
    }
    if use_cd:
        result["cd_var"] = float(cov_diag[9*N + 3*N_uma + 3])
    return result


def compute_covariance(context: dict) -> dict:
    """
    Compute covariance from a context captured by build_and_solve(return_covariance_context=True).

    Performs the Jacobian evaluation and SVD without re-running IPOPT.
    """
    cov_start = time.perf_counter()
    cov = _compute_covariance(
        context["cost_residuals"],
        context["dyn_constraint_syms"],
        context["sol"],
        context["r"], context["v"], context["q"], context["a"], context["b"],
        context["N"],
        cd=context.get("cd"),
    )
    covariance_time_ms = (time.perf_counter() - cov_start) * 1000.0
    print(f"[covariance] Complete in {covariance_time_ms:.1f} ms")
    return cov


def compute_outlier_residuals(
    result: dict,
    landmark_measurements:  np.ndarray,         # full original (N_total, 7)
    landmark_uncertainties: np.ndarray | None,  # full original (N_total,) or None
) -> np.ndarray:
    """
    Evaluate landmark residuals for rejected measurements against the final solution.

    Uses the CasADi landmark_residual function with the converged positions and
    quaternions, so the geometry is identical to what was used during the solve.

    Returns
    -------
    outlier_res : (N_outliers, 3) float64
        σ-normalised residual vectors, in the same order as the rejected rows
        of landmark_measurements (i.e. where lmk_outlier_flags == 1).
    """
    flags    = np.asarray(result["lmk_outlier_flags"])
    outlier_mask = flags == 1
    if not outlier_mask.any():
        return np.zeros((0, 3))

    outlier_meas = landmark_measurements[outlier_mask]
    outlier_unc  = (landmark_uncertainties[outlier_mask]
                    if landmark_uncertainties is not None
                    else np.full(outlier_mask.sum(), LANDMARK_STD))

    ts_final   = np.asarray(result["state_timestamps"])
    pos_final  = result["positions"]    # (N, 3)
    quat_final = result["quaternions"]  # (N, 4) [x,y,z,w]
    M = len(ts_final)

    outlier_res = np.zeros((outlier_mask.sum(), 3))
    for j, (meas, sigma) in enumerate(zip(outlier_meas, outlier_unc)):
        t_lmk = float(meas[0])
        lo = int(np.searchsorted(ts_final, t_lmk))
        if lo == 0:
            idx = 0
        elif lo == M:
            idx = M - 1
        else:
            idx = lo if ts_final[lo] - t_lmk < t_lmk - ts_final[lo - 1] else lo - 1

        res = landmark_residual(
            pos_final[idx],   # (3,) → ca.DM
            quat_final[idx],  # (4,) → ca.DM
            meas[4:7],        # landmark ECI position
            meas[1:4],        # measured bearing
            float(sigma),
        )
        outlier_res[j] = np.asarray(res).ravel()

    return outlier_res


# ── Main solver ─────────────────────────────────────────────────────────────────

def build_and_solve(
    landmark_measurements:  np.ndarray,         # (N_lmk, 7): [t_j2000, bx, by, bz, ex, ey, ez]
    landmark_group_starts:  np.ndarray,         # (N_lmk,) bool
    gyro_measurements:      np.ndarray,         # (N_gyro, 4): [t_j2000, wx, wy, wz] rad/s
    landmark_uncertainties: np.ndarray | None = None,  # (N_lmk,) per-measurement sigma [rad]
    uma_std:         float          = UMA_STD_DEV,
    integrator_type: IntegratorType = IntegratorType.FORWARD_EULER,
    use_j2:           bool        = False,
    use_drag:         bool        = False,
    cd_nominal:       float       = 2.2,
    cd_std:       float | None = None,
    ipopt_opts:       dict | None  = None,
    compute_covariance: bool       = True,
    warm_start:         dict | None = None,
    landmark_huber_M:   float      = 3.0,
    return_covariance_context: bool = False,
) -> dict:
    """
    Build and solve the constrained fixed-bias batch OD problem.

    Linear dynamics are enforced as IPOPT equality constraints; the unmodelled
    acceleration a[i] absorbs any model error and is penalised by a Gaussian
    prior with standard deviation uma_std [km/s²].

    Parameters
    ----------
    integrator_type : IntegratorType
        FORWARD_EULER (default) or RK4.
    use_j2 : bool
        Include J2 zonal harmonic gravity perturbation in the dynamics model.
    use_drag : bool
        Include exponential-density atmospheric drag in the dynamics model.
        When True, cd_nominal and cd_std must be provided.
    cd_nominal : float
        Prior mean / initial guess for Cd  [-].
        Unit conversion: 1 m²/kg = 1×10⁻⁶ km²/kg.
        Assumed for argus CubeSat (Cd=2.2, A=0.01 m², m=1.3 kg): cd ≈ 2.2.
    cd_std : float | None
        Prior standard deviation for cd  [-].  Required when use_drag=True.
    compute_covariance : bool
        When True, compute covariance inline after solving (step 8).
        Mutually exclusive with return_covariance_context.
    return_covariance_context : bool
        When True, include a ``_covariance_context`` key in the result dict
        containing all objects needed to call ``compute_covariance()`` later,
        without re-running IPOPT.  Use this in the outlier-rejection loop so
        the caller can defer the (expensive) Jacobian evaluation until
        convergence is confirmed.
    warm_start : dict | None
        If provided, use this result dict (from a previous build_and_solve call)
        as the initial guess instead of running TrajectoryInitializer.  The dict
        must contain positions, velocities, quaternions, uma_accelerations, and
        gyro_bias keys (and cd when drag is enabled).

    Returns a dict:
        state_timestamps    : (N,)      float64
        positions           : (N, 3)    float64  [km]
        velocities          : (N, 3)    float64  [km/s]
        quaternions         : (N, 4)    float64  [x,y,z,w]
        uma_accelerations   : (N-1, 3)  float64  [km/s²]
        gyro_bias           : (3,)      float64  [rad/s]
        residuals           : (flat,)   float64  (see process_residuals format)
        pos_var / vel_var / rot_var / uma_var / bias_var  — covariance blocks
    """
    # ── 1. Timeline (mirrors C++: states = gyro timestamps) ───────────────────
    ts_list, lmk_group_indices = get_state_timestamps(
        landmark_measurements, landmark_group_starts, gyro_measurements
    )
    ts    = np.array(ts_list, dtype=np.float64)
    N     = len(ts)
    N_uma = N - 1

    # Every state i has exactly one gyro measurement (state timestamps = gyro timestamps)
    gyro_at_state: dict[int, int] = {i: i for i in range(N)}
    lmk_groups = _build_landmark_groups(landmark_group_starts, lmk_group_indices)

    if use_drag and cd_std is None:
        raise ValueError("cd_std must be provided when use_drag=True")

    physics_flags = (
        ("J2 " if use_j2 else "") +
        (f"drag(cd₀={cd_nominal:.2e} ±{cd_std:.2e}) " if use_drag else "") or
        "2-body"
    ).strip()
    print(f"[optimizer] {N} states | "
          f"{len(lmk_group_indices)} landmark groups | "
          f"span {ts[-1] - ts[0]:.1f} s  |  UMA σ = {uma_std:.2e} km/s²  |  "
          f"integrator = {integrator_type.value}  |  physics = {physics_flags}")

    # ── 2. Decision variables ──────────────────────────────────────────────────
    opti = ca.Opti()

    r      = opti.variable(3, N)      # positions   [km]
    v      = opti.variable(3, N)      # velocities  [km/s]
    q      = opti.variable(4, N)      # quaternions [x,y,z,w]
    a      = opti.variable(3, N_uma)  # UMA per interval [km/s²]
    b      = opti.variable(3)         # fixed gyro bias [rad/s]
    cd     = opti.variable(1) if use_drag else None  # drag coeff [-]

    dynamics = OrbitalDynamics(use_j2=use_j2, use_drag=use_drag,
                               cd=cd if use_drag else 0.0)

    # ── 3. Equality constraints ────────────────────────────────────────────────
    # Quaternion unit norm
    for i in range(N):
        opti.subject_to(ca.dot(q[:, i], q[:, i]) == 1.0)

    # Linear dynamics (hard): store constraint expressions for residuals / covariance
    dyn_constraint_syms: list[ca.MX] = []
    for i in range(N_uma):
        t0 = float(ts[i])
        dt = float(ts[i + 1] - ts[i])
        c6 = linear_dynamics_constraint(r[:, i], v[:, i], r[:, i + 1], v[:, i + 1],
                                         a[:, i], t0, dt, integrator_type, dynamics)
        opti.subject_to(c6 == 0)
        dyn_constraint_syms.append(c6)

    # ── 4. Soft cost ───────────────────────────────────────────────────────────
    obj                                  = ca.MX.zeros(1)
    angular_res_syms: list[ca.MX | None] = []
    landmark_res_syms: list[ca.MX]       = []
    uma_res_syms: list[ca.MX]            = []

    # — UMA prior —
    for i in range(N_uma):
        res = uma_prior_residual(a[:, i], uma_std)
        uma_res_syms.append(res)
        obj += ca.dot(res, res)

    # — Drag cd prior —
    cd_res_sym: ca.MX | None = None
    if use_drag:
        cd_res_sym = drag_prior_residual(cd, cd_nominal, cd_std)
        obj += cd_res_sym ** 2

    # — Angular dynamics (fixed bias) —
    for i in range(N_uma):
        if i not in gyro_at_state:
            angular_res_syms.append(None)
            continue
        j        = gyro_at_state[i]
        dt       = float(ts[i + 1] - ts[i])
        quat_std = GYRO_WN_STD_DEV * dt
        gyro_w   = ca.DM(gyro_measurements[j, 1:4].astype(float))
        res = angular_dynamics_residual_fix_bias(q[:, i], q[:, i + 1], gyro_w, b, dt, quat_std)
        angular_res_syms.append(res)
        obj += ca.dot(res, res)

    # — Landmark bearing —
    for state_idx, meas_rows in lmk_groups:
        for row in meas_rows:
            lmk_pos      = ca.DM(landmark_measurements[row, 4:7].astype(float))
            bearing_meas = ca.DM(landmark_measurements[row, 1:4].astype(float))
            sigma = (float(landmark_uncertainties[row])
                     if landmark_uncertainties is not None else LANDMARK_STD)
            res = landmark_residual(r[:, state_idx], q[:, state_idx],
                                    lmk_pos, bearing_meas, sigma)
            landmark_res_syms.append(res)
            obj += pseudo_huber_cost(res, landmark_huber_M)

    opti.minimize(obj)

    # ── 5. Initial guess ──────────────────────────────────────────────────────
    if warm_start is not None:
        init_positions   = warm_start["positions"]
        init_velocities  = warm_start["velocities"]
        init_quaternions = warm_start["quaternions"]
        init_gyro_bias   = np.asarray(warm_start["gyro_bias"]).ravel()
        for i in range(N):
            opti.set_initial(r[:, i], init_positions[i])
            opti.set_initial(v[:, i], init_velocities[i])
            opti.set_initial(q[:, i], init_quaternions[i])
        for i in range(N_uma):
            opti.set_initial(a[:, i], warm_start["uma_accelerations"][i])
        opti.set_initial(b, init_gyro_bias)
        if use_drag:
            opti.set_initial(cd, warm_start.get("cd", cd_nominal))
    else:
        traj_init = TrajectoryInitializer(
            ts_list, landmark_measurements, landmark_group_starts,
            gyro_measurements, estimate_bias=True,
        )
        init_positions   = traj_init.positions
        init_velocities  = traj_init.velocities
        init_quaternions = traj_init.quaternions
        init_gyro_bias   = traj_init.gyro_bias
        for i in range(N):
            opti.set_initial(r[:, i], init_positions[i])
            opti.set_initial(v[:, i], init_velocities[i])
            opti.set_initial(q[:, i], init_quaternions[i])
        for i in range(N_uma):
            opti.set_initial(a[:, i], [0.0, 0.0, 0.0])
        opti.set_initial(b, init_gyro_bias)
        if use_drag:
            opti.set_initial(cd, cd_nominal)

    # ── 6. IPOPT ───────────────────────────────────────────────────────────────
    opts: dict = {
        "ipopt.max_iter":    1000,
        "ipopt.tol":         1e-6,
        "ipopt.print_level": 5,
    }
    if ipopt_opts:
        opts.update(ipopt_opts)

    opti.solver("ipopt", opts)
    print("[optimizer] Starting IPOPT solve …")
    sol = opti.solve()
    print("[optimizer] Solve complete.")

    # ── 7. Evaluate residuals ─────────────────────────────────────────────────
    # Format: (N-1)*6 linear  |  (N-1)*3 angular  |  N_lmk*3 landmark
    # Linear dynamics residuals are exactly zero (hard constraints); evaluate
    # the raw constraint violation for verification.
    lin_vals = np.concatenate(
        [np.asarray(sol.value(c)).ravel() for c in dyn_constraint_syms]
    )
    ang_vals = np.concatenate([
        np.asarray(sol.value(res)).ravel() if res is not None else np.zeros(3)
        for res in angular_res_syms
    ])
    lmk_vals = np.concatenate(
        [np.asarray(sol.value(res)).ravel() for res in landmark_res_syms]
    )
    # ── 8. Covariance ─────────────────────────────────────────────────────────
    cov: dict = {}
    cost_residuals = (
        uma_res_syms
        + ([cd_res_sym] if cd_res_sym is not None else [])
        + [res for res in angular_res_syms if res is not None]
        + landmark_res_syms
    )
    if compute_covariance:
        cov_start = time.perf_counter()
        cov = _compute_covariance(cost_residuals, dyn_constraint_syms, sol,
                                  r, v, q, a, b, N,
                                  cd=cd if use_drag else None)
        covariance_time_ms = (time.perf_counter() - cov_start) * 1000.0
        print(f"[covariance] Complete in {covariance_time_ms:.1f} ms")

    # ── 9. Extract solution ────────────────────────────────────────────────────
    a_val = np.asarray(sol.value(a))
    if a_val.ndim == 1:
        a_val = a_val.reshape(3, 1)

    result = {
        "state_timestamps":   ts,
        "positions":          sol.value(r).T,           # (N, 3)
        "velocities":         sol.value(v).T,           # (N, 3)
        "quaternions":        sol.value(q).T,           # (N, 4)  [x,y,z,w]
        "gyro_bias":          np.asarray(sol.value(b)), # (3,)
        "uma_accelerations":  a_val.T,                  # (N-1, 3)
        # initial trajectory for CSV output (warm_start values or cold-start)
        "init_positions":     init_positions,           # (N, 3)
        "init_velocities":    init_velocities,          # (N, 3)
        "init_quaternions":   init_quaternions,         # (N, 4)
        "init_gyro_bias":     init_gyro_bias,           # (3,)
        # residuals split by type (for separate CSV files)
        "lin_residuals":      lin_vals.reshape(-1, 6),  # (N-1, 6)
        "ang_residuals":      ang_vals.reshape(-1, 3),  # (N-1, 3)
        "lmk_residuals":      lmk_vals.reshape(-1, 3),  # (N_lmk, 3)
        **cov,                                          # empty dict when compute_covariance=False
    }
    if use_drag:
        result["cd"] = float(sol.value(cd))

    if return_covariance_context:
        result["_covariance_context"] = {
            "cost_residuals":     cost_residuals,
            "dyn_constraint_syms": dyn_constraint_syms,
            "sol":                sol,
            "r": r, "v": v, "q": q, "a": a, "b": b,
            "N":                  N,
            "cd":                 cd if use_drag else None,
        }

    return result


def build_and_solve_with_outlier_rejection(
    landmark_measurements:  np.ndarray,
    landmark_group_starts:  np.ndarray,
    gyro_measurements:      np.ndarray,
    landmark_uncertainties: np.ndarray | None = None,
    uma_std:         float          = UMA_STD_DEV,
    integrator_type: IntegratorType = IntegratorType.FORWARD_EULER,
    use_j2:           bool        = False,
    use_drag:         bool        = False,
    cd_nominal:       float       = 2.2,
    cd_std:       float | None = None,
    ipopt_opts:       dict | None  = None,
    compute_covariance: bool       = True,
    mahal_threshold:    float      = 5.0,
    max_iterations:     int        = 10,
    landmark_huber_M:   float      = 3.0,
) -> dict:
    """
    Iterative batch OD with per-landmark Mahalanobis-distance outlier rejection.

    Algorithm
    ---------
    1. Solve with all measurements (cold-start).
    2. Compute Mahalanobis distance for each landmark row:
           d_i = ‖lmk_residuals[i]‖₂
       The residuals are already σ-normalised inside landmark_residual(), so d_i
       is the Mahalanobis distance directly.
    3. Reject rows with d_i > mahal_threshold (individual landmarks, not groups).
    4. Warm-start the next solve from the current solution and repeat from (2).
    5. Stop when no rows are rejected (converged) or max_iterations is reached.
    6. If compute_covariance=True, run one final warm-started solve to compute
       the covariance on the converged, outlier-free measurement set.

    The cold-start initial trajectory (from TrajectoryInitializer) is preserved
    in the returned result's init_* fields regardless of how many iterations ran,
    so the output CSV matches what a single-shot solve would have produced.

    Parameters
    ----------
    mahal_threshold : float
        Per-measurement rejection threshold in normalised residual units.
        Default 5.0 is intentionally conservative — only egregious outliers
        are rejected on the first pass.
    max_iterations : int
        Safety cap on the number of outlier-rejection iterations (excludes the
        final covariance solve).
    """
    active_meas   = landmark_measurements.copy()
    active_starts = landmark_group_starts.copy()
    active_unc    = landmark_uncertainties.copy() if landmark_uncertainties is not None else None

    N_total = len(landmark_measurements)
    outlier_flags           = np.zeros(N_total, dtype=np.int8)
    active_original_indices = np.arange(N_total)

    warm_start:     dict | None = None
    preserved_init: dict | None = None   # cold-start init trajectory from iteration 0

    for iteration in range(max_iterations):
        n_active_groups = int(active_starts.sum())
        print(f"\n[outlier_rejection] Iteration {iteration + 1} | "
              f"{n_active_groups} groups | {len(active_meas)} measurements")

        result = build_and_solve(
            active_meas, active_starts, gyro_measurements,
            landmark_uncertainties     = active_unc,
            uma_std                    = uma_std,
            integrator_type            = integrator_type,
            use_j2                     = use_j2,
            use_drag                   = use_drag,
            cd_nominal                 = cd_nominal,
            cd_std                     = cd_std,
            ipopt_opts                 = ipopt_opts,
            compute_covariance         = False,
            warm_start                 = warm_start,
            landmark_huber_M           = landmark_huber_M,
            return_covariance_context  = compute_covariance,
        )

        # Preserve the cold-start init trajectory for final result reporting
        if preserved_init is None:
            preserved_init = {
                "init_positions":  result["init_positions"],
                "init_velocities": result["init_velocities"],
                "init_quaternions": result["init_quaternions"],
                "init_gyro_bias":  result["init_gyro_bias"],
            }

        # Mahalanobis distance: residuals already normalised by σ, so ‖res‖ = d_mahal
        lmk_res    = result["lmk_residuals"]           # (N_active, 3)
        mahal_dist = np.linalg.norm(lmk_res, axis=1)   # (N_active,)

        keep_mask  = mahal_dist <= mahal_threshold
        n_rejected = int((~keep_mask).sum())

        print(f"[outlier_rejection] Mahalanobis distances — "
              f"max={mahal_dist.max():.3f}  mean={mahal_dist.mean():.3f}  "
              f"rejected={n_rejected}/{len(active_meas)} (threshold={mahal_threshold:.2f}σ)")

        if n_rejected == 0:
            print(f"[outlier_rejection] Converged after {iteration + 1} iteration(s).")
            if compute_covariance:
                print("[outlier_rejection] Computing covariance from converged solution …")
                ctx = result.pop("_covariance_context")
                cov_start = time.perf_counter()
                cov_dict = _compute_covariance(
                    ctx["cost_residuals"], ctx["dyn_constraint_syms"],
                    ctx["sol"], ctx["r"], ctx["v"], ctx["q"], ctx["a"], ctx["b"],
                    ctx["N"], cd=ctx.get("cd"),
                )
                print(f"[covariance] Complete in {(time.perf_counter() - cov_start) * 1000:.1f} ms")
                result.update(cov_dict)
            break

        if iteration == max_iterations - 1:
            print(f"[outlier_rejection] Warning: reached max_iterations={max_iterations}; "
                  f"{n_rejected} outlier(s) remain.")
            result.pop("_covariance_context", None)
            break

        result.pop("_covariance_context", None)
        outlier_flags[active_original_indices[~keep_mask]] = 1
        active_original_indices = active_original_indices[keep_mask]
        active_meas, active_starts, active_unc = _filter_landmark_measurements(
            active_meas, active_starts, active_unc, keep_mask
        )
        warm_start = result

    result["lmk_outlier_flags"] = outlier_flags
    result["n_od_iterations"]   = iteration + 1

    result["lmk_outlier_residuals"] = compute_outlier_residuals(
        result, landmark_measurements, landmark_uncertainties
    )

    # Restore cold-start init trajectory so output CSVs are consistent
    if preserved_init is not None:
        result.update(preserved_init)

    return result


# ── Output (CSV format matching C++ run_od_on_dataset) ───────────────────────────

_STATE_HEADER = (
    "timestamp_j2000,pos_x_km,pos_y_km,pos_z_km,"
    "vel_x_kms,vel_y_kms,vel_z_kms,"
    "quat_x,quat_y,quat_z,quat_w"
)
_DYN_RES_HEADER = (
    "timestamp_j2000,pos_res_x,pos_res_y,pos_res_z,"
    "vel_res_x,vel_res_y,vel_res_z,"
    "rot_res_x,rot_res_y,rot_res_z,"
    "gyro_bias_res_x,gyro_bias_res_y,gyro_bias_res_z"
)
_LMK_RES_HEADER = "res_x,res_y,res_z,outlier"
_COV_HEADER = (
    "timestamp_j2000,pos_cov_x,pos_cov_y,pos_cov_z,"
    "vel_cov_x,vel_cov_y,vel_cov_z,"
    "rot_cov_x,rot_cov_y,rot_cov_z"
)


def _write_state_csv(path: Path, timestamps: np.ndarray,
                     positions: np.ndarray, velocities: np.ndarray,
                     quaternions: np.ndarray) -> None:
    N = len(timestamps)
    with open(path, "w", newline="") as f:
        f.write(_STATE_HEADER + "\n")
        for i in range(N):
            row = [timestamps[i],
                   positions[i, 0], positions[i, 1], positions[i, 2],
                   velocities[i, 0], velocities[i, 1], velocities[i, 2],
                   quaternions[i, 0], quaternions[i, 1],
                   quaternions[i, 2], quaternions[i, 3]]
            f.write(",".join(f"{v:.12g}" for v in row) + "\n")


def save_results(results: dict, results_dir: Path, meta: dict | None = None) -> None:
    """
    Write optimizer results to CSVs matching C++ run_od_on_dataset output.

    Files written to results_dir/:
      state_estimates.csv       — N×14 final state estimates
      initial_trajectory.csv    — N×14 initial trajectory guess
      dynamics_residuals.csv    — (N-1)×13 (timestamp + 12 residual fields)
      landmark_residuals.csv    — N_lmk×3
      covariance.csv            — N×13  (if covariance was computed)
      od_result.json            — metadata (if meta dict is provided)
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ts   = results["state_timestamps"]
    bias = np.asarray(results["gyro_bias"]).ravel()
    N    = len(ts)

    # state_estimates.csv  (11 cols; gyro_bias moved to od_result.json estimates)
    _write_state_csv(
        results_dir / "state_estimates.csv",
        ts, results["positions"], results["velocities"],
        results["quaternions"],
    )

    # initial_trajectory.csv
    _write_state_csv(
        results_dir / "initial_trajectory.csv",
        ts, results["init_positions"], results["init_velocities"],
        results["init_quaternions"],
    )

    # dynamics_residuals.csv
    # lin_residuals (N-1, 6): pos+vel; ang_residuals (N-1, 3): rot; zeros for gyro_bias_res
    lin_res = np.asarray(results["lin_residuals"])   # (N-1, 6)
    ang_res = np.asarray(results["ang_residuals"])   # (N-1, 3)
    with open(results_dir / "dynamics_residuals.csv", "w", newline="") as f:
        f.write(_DYN_RES_HEADER + "\n")
        for i in range(N - 1):
            row = [ts[i],
                   lin_res[i, 0], lin_res[i, 1], lin_res[i, 2],
                   lin_res[i, 3], lin_res[i, 4], lin_res[i, 5],
                   ang_res[i, 0], ang_res[i, 1], ang_res[i, 2],
                   0.0, 0.0, 0.0]
            f.write(",".join(f"{v:.12g}" for v in row) + "\n")

    # landmark_residuals.csv
    lmk_res = np.asarray(results["lmk_residuals"])  # (N_active, 3)
    if "lmk_outlier_flags" in results:
        flags = np.asarray(results["lmk_outlier_flags"])  # (N_total,)
        N_total = len(flags)
        lmk_res_full = np.full((N_total, 3), np.nan)
        lmk_res_full[flags == 0] = lmk_res
        if "lmk_outlier_residuals" in results:
            lmk_res_full[flags == 1] = np.asarray(results["lmk_outlier_residuals"])
    else:
        flags = np.zeros(len(lmk_res), dtype=np.int8)
        lmk_res_full = lmk_res
    with open(results_dir / "landmark_residuals.csv", "w", newline="") as f:
        f.write(_LMK_RES_HEADER + "\n")
        for i in range(len(lmk_res_full)):
            rx, ry, rz = lmk_res_full[i]
            flag = int(flags[i])
            res_str = (
                f"{rx:.12g},{ry:.12g},{rz:.12g}"
                if not np.isnan(rx) else
                "nan,nan,nan"
            )
            f.write(f"{res_str},{flag}\n")

    # covariance.csv (10 cols; gyro_bias_cov moved to od_result.json estimates)
    if "pos_var" in results:
        pos_var = np.asarray(results["pos_var"])
        vel_var = np.asarray(results["vel_var"])
        rot_var = np.asarray(results["rot_var"])
        with open(results_dir / "covariance.csv", "w", newline="") as f:
            f.write(_COV_HEADER + "\n")
            for i in range(N):
                row = [ts[i],
                       pos_var[i, 0], pos_var[i, 1], pos_var[i, 2],
                       vel_var[i, 0], vel_var[i, 1], vel_var[i, 2],
                       rot_var[i, 0], rot_var[i, 1], rot_var[i, 2]]
                f.write(",".join(f"{v:.12g}" for v in row) + "\n")

    # Populate meta["estimates"] with gyro_bias, cd, and their covariances
    if meta is not None:
        outputs = meta.setdefault("outputs", {})
        outputs["covariance_available"] = "pos_var" in results
        outputs["covariance_computed"] = "pos_var" in results
        if "n_od_iterations" in results:
            outputs["od_iterations"] = int(results["n_od_iterations"])
        if "lmk_outlier_flags" in results:
            outputs["landmarks_rejected"] = int(np.asarray(results["lmk_outlier_flags"]).sum())

        est = meta.setdefault("estimates", {})
        est["gyro_bias_x_rads"] = float(bias[0])
        est["gyro_bias_y_rads"] = float(bias[1])
        est["gyro_bias_z_rads"] = float(bias[2])
        if "cd" in results:
            est["cd"] = float(results["cd"])
        if "bias_var" in results:
            bv = np.asarray(results["bias_var"]).ravel()
            est["gyro_bias_cov_x_rads2"] = float(bv[0])
            est["gyro_bias_cov_y_rads2"] = float(bv[1])
            est["gyro_bias_cov_z_rads2"] = float(bv[2])
        if "cd_var" in results:
            est["cd_var"] = float(results["cd_var"])

    # od_result.json
    if meta is not None:
        with open(results_dir / "od_result.json", "w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")

    print(f"[optimizer] Results saved → {results_dir}")
