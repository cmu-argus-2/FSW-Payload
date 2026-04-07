"""
CasADi/IPOPT batch orbit determination optimizer — fixed gyro bias, constrained dynamics.

Decision variables:
  r[i]  — ECI position        (3,)  [km]        per state
  v[i]  — ECI velocity        (3,)  [km/s]      per state
  q[i]  — body-to-ECI quat    (4,)  [x,y,z,w]  per state  (||q||=1 constraint)
  a[i]  — unmodelled accel    (3,)  [km/s²]     per interval
  b     — fixed gyro bias     (3,)  [rad/s]     global

Equality constraints (IPOPT enforces these to machine precision):
  linear dynamics:  r_{i+1} = r_i + v_i dt + ½ a_i dt²
                    v_{i+1} = v_i + (a_kepler(r_i) + a_i) dt
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

from pathlib import Path

import casadi as ca
import h5py
import numpy as np
import scipy.linalg

from data_loader import get_state_timestamps
from residuals import (
    IntegratorType,
    angular_dynamics_residual_fix_bias,
    keplerian_accel,
    landmark_residual,
    linear_dynamics_constraint,
    uma_prior_residual,
)

# ── Noise / prior parameters ───────────────────────────────────────────────────
UMA_STD_DEV     = 1e-5        # km/s²  — expected magnitude of unmodelled forces
GYRO_WN_STD_DEV = 0.0008726   # rad/s  — gyro white noise std dev
LANDMARK_STD    = 0.009       # rad    — landmark bearing noise

# ── Initial guess (circular orbit ~7000 km altitude, no spin, zero bias) ───────
INIT_R = np.array([0.0, 0.0, 7000.0])    # km
INIT_V = np.array([0.0, 8.0, 0.0])       # km/s
INIT_Q = np.array([0.0, 0.0, 0.0, 1.0]) # identity [x,y,z,w]
INIT_B = np.array([0.0, 0.0, 0.0])       # rad/s


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


def _build_tangent_projection(q_vals: np.ndarray, N: int, N_uma: int) -> np.ndarray:
    """
    Build the (n_full) × (n_reduced) tangent-space projection matrix T.

    Full parameter vector  x = [vec(r); vec(v); vec(q); vec(a); b]
    sizes:                      3N       3N       4N      3·N_uma  3
    n_full = 10N + 3·N_uma + 3

    Reduced (tangent) vector x_t = [vec(r); vec(v); vec(q_t); vec(a); b]
    sizes:                           3N       3N       3N       3·N_uma  3
    n_reduced = 9N + 3·N_uma + 3

    Only the quaternion blocks need non-trivial projection (4→3 per state).
    All other blocks are identity.
    """
    n_full    = 10 * N + 3 * N_uma + 3
    n_reduced =  9 * N + 3 * N_uma + 3
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
    T[a_row + 3*N_uma:, a_col + 3*N_uma:] = np.eye(3)

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
) -> dict[str, np.ndarray]:
    """
    Approximate covariance from the augmented Jacobian at the solution.

    For a constrained least-squares problem the covariance comes from:
        Σ ≈ (J_aug^T J_aug)^{-1}
    where J_aug stacks:
      • The Jacobian of all soft cost residuals (angular, landmark, UMA prior)
      • The Jacobian of the dynamics equality constraints, scaled by a tight
        normalisation factor — this captures the constraint coupling between
        states and UMA without needing to form the full KKT matrix.

    Quaternion blocks are projected onto their 3D tangent space.

    Returns
    -------
    dict with:
        pos_var  : (N, 3)       position variance        [km²]
        vel_var  : (N, 3)       velocity variance        [(km/s)²]
        rot_var  : (N, 3)       attitude tangent variance [rad²]
        uma_var  : (N_uma, 3)   UMA variance             [(km/s²)²]
        bias_var : (3,)         gyro bias variance       [(rad/s)²]
    """
    print("[covariance] Building symbolic Jacobian …")

    N_uma = N - 1
    # Full symbolic parameter vector (column-major vectorisation)
    x_sym = ca.vertcat(ca.vec(r), ca.vec(v), ca.vec(q), ca.vec(a), b)

    # Dynamics constraints are included with a tight normalisation so their
    # Jacobian propagates the state-UMA coupling into the covariance.
    # The normalisation cancels out of (J^T J)^{-1} for well-constrained
    # directions, leaving only the information from the soft residuals.
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

    print("[covariance] Evaluating Jacobian at solution …")
    J_val = np.array(J_fn(x_val))

    print("[covariance] Projecting to tangent space …")
    T = _build_tangent_projection(q_val.T, N, N_uma)
    J_reduced = J_val @ T

    # Diagonal of (J^T J)^{-1} via SVD — avoids forming J^T J explicitly
    print("[covariance] Computing SVD …")
    _, S, Vt = scipy.linalg.svd(J_reduced, full_matrices=False)

    tol      = S[0] * max(J_reduced.shape) * np.finfo(float).eps * 1e3
    S_safe   = np.where(S > tol, S, np.inf)
    cov_diag = np.sum((Vt.T / S_safe) ** 2, axis=1)

    n_reduced = 9 * N + 3 * N_uma + 3
    pos_var  = cov_diag[             :  3*N         ].reshape(N,     3, order="F")
    vel_var  = cov_diag[ 3*N         :  6*N         ].reshape(N,     3, order="F")
    rot_var  = cov_diag[ 6*N         :  9*N         ].reshape(N,     3, order="F")
    uma_var  = cov_diag[ 9*N         :  9*N+3*N_uma ].reshape(N_uma, 3, order="F")
    bias_var = cov_diag[ 9*N+3*N_uma :               ]

    return {
        "pos_var":  pos_var,
        "vel_var":  vel_var,
        "rot_var":  rot_var,
        "uma_var":  uma_var,
        "bias_var": bias_var,
    }


# ── Main solver ─────────────────────────────────────────────────────────────────

def build_and_solve(
    landmark_measurements:  np.ndarray,   # (N_lmk,  7): [t, bx, by, bz, lx, ly, lz]
    landmark_group_starts:  np.ndarray,   # (N_lmk,)  bool
    gyro_measurements:      np.ndarray,   # (N_gyro,  4): [t, wx, wy, wz]
    max_dt:          float          = 60.0,
    uma_std:         float          = UMA_STD_DEV,
    integrator_type: IntegratorType = IntegratorType.FORWARD_EULER,
    ipopt_opts:      dict | None    = None,
) -> dict:
    """
    Build and solve the constrained fixed-bias batch OD problem.

    Linear dynamics are enforced as IPOPT equality constraints; the unmodelled
    acceleration a[i] absorbs any model error and is penalised by a Gaussian
    prior with standard deviation uma_std [km/s²].

    Parameters
    ----------
    integrator_type : IntegratorType

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
    # ── 1. Timeline ────────────────────────────────────────────────────────────
    ts_list, lmk_group_indices, gyro_indices = get_state_timestamps(
        landmark_measurements, landmark_group_starts, gyro_measurements, max_dt
    )
    ts    = np.array(ts_list, dtype=np.float64)
    N     = len(ts)
    N_uma = N - 1

    gyro_at_state: dict[int, int] = {si: j for j, si in enumerate(gyro_indices)}
    lmk_groups = _build_landmark_groups(landmark_group_starts, lmk_group_indices)

    print(f"[optimizer] {N} states | "
          f"{len(lmk_group_indices)} landmark groups | "
          f"{len(gyro_indices)} gyro measurements | "
          f"span {ts[-1] - ts[0]:.1f} s  |  UMA σ = {uma_std:.2e} km/s²  |  "
          f"integrator = {integrator_type.value}")

    # ── 2. Decision variables ──────────────────────────────────────────────────
    opti = ca.Opti()

    r = opti.variable(3, N)      # positions   [km]
    v = opti.variable(3, N)      # velocities  [km/s]
    q = opti.variable(4, N)      # quaternions [x,y,z,w]
    a = opti.variable(3, N_uma)  # UMA per interval [km/s²]
    b = opti.variable(3)         # fixed gyro bias [rad/s]

    # ── 3. Equality constraints ────────────────────────────────────────────────
    # Quaternion unit norm
    for i in range(N):
        opti.subject_to(ca.dot(q[:, i], q[:, i]) == 1.0)

    # Linear dynamics (hard): store constraint expressions for residuals / covariance
    dyn_constraint_syms: list[ca.MX] = []
    for i in range(N_uma):
        dt  = float(ts[i + 1] - ts[i])
        c6  = linear_dynamics_constraint(r[:, i], v[:, i], r[:, i + 1], v[:, i + 1],
                                          a[:, i], dt, integrator_type)
        opti.subject_to(c6 == 0)
        dyn_constraint_syms.append(c6)

    # ── 4. Soft cost ───────────────────────────────────────────────────────────
    obj                    = ca.MX.zeros(1)
    angular_res_syms: list[ca.MX | None] = []
    landmark_res_syms: list[ca.MX]       = []
    uma_res_syms: list[ca.MX]            = []

    # — UMA prior —
    for i in range(N_uma):
        res = uma_prior_residual(a[:, i], uma_std)
        uma_res_syms.append(res)
        obj += ca.dot(res, res)

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
            res = landmark_residual(r[:, state_idx], q[:, state_idx],
                                    lmk_pos, bearing_meas, LANDMARK_STD)
            landmark_res_syms.append(res)
            obj += ca.dot(res, res)

    opti.minimize(obj)

    # ── 5. Initial guess ───────────────────────────────────────────────────────
    for i in range(N):
        opti.set_initial(r[:, i], INIT_R)
        opti.set_initial(v[:, i], INIT_V)
        opti.set_initial(q[:, i], INIT_Q)
    for i in range(N_uma):
        opti.set_initial(a[:, i], [0.0, 0.0, 0.0])
    opti.set_initial(b, INIT_B)

    # ── 6. IPOPT ───────────────────────────────────────────────────────────────
    opts: dict = {
        "ipopt.max_iter":    10000,
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
    residuals_flat = np.concatenate([lin_vals, ang_vals, lmk_vals])

    # ── 8. Covariance ─────────────────────────────────────────────────────────
    cost_residuals = uma_res_syms + [r for r in angular_res_syms if r is not None] + landmark_res_syms
    cov = _compute_covariance(cost_residuals, dyn_constraint_syms, sol, r, v, q, a, b, N)

    # ── 9. Extract solution ────────────────────────────────────────────────────
    a_val = np.asarray(sol.value(a))
    if a_val.ndim == 1:
        a_val = a_val.reshape(3, 1)

    return {
        "state_timestamps":  ts,
        "positions":         sol.value(r).T,    # (N, 3)
        "velocities":        sol.value(v).T,    # (N, 3)
        "quaternions":       sol.value(q).T,    # (N, 4)
        "uma_accelerations": a_val.T,           # (N-1, 3)
        "gyro_bias":         sol.value(b),      # (3,)
        "residuals":         residuals_flat,
        **cov,
    }


# ── Output ──────────────────────────────────────────────────────────────────────

def save_results(results: dict, output_path: Path) -> None:
    """
    Write optimizer results to HDF5.

    state_estimates (N, 14):
        [timestamp, px, py, pz, vx, vy, vz, qx, qy, qz, qw, bx, by, bz]

    residuals ((N-1)*6 + (N-1)*3 + N_lmk*3,):
        (N-1)*6  linear dynamics constraint violations  (should be ~0)
        (N-1)*3  angular dynamics residuals
        N_lmk*3  landmark bearing residuals

    state_estimate_covariance_diagonal (9N+3,) — fix_bias layout:
        state 0   : [px, py, pz, vx, vy, vz, rx, ry, rz, bx, by, bz]  (12 values)
        state 1…N : [px, py, pz, vx, vy, vz, rx, ry, rz]               (9 values each)

    uma_accelerations (N-1, 3):  estimated unmodelled acceleration [km/s²]
    uma_variance      (N-1, 3):  UMA marginal variance             [(km/s²)²]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    N    = len(results["state_timestamps"])
    bias = np.asarray(results["gyro_bias"]).ravel()

    state_estimates          = np.zeros((N, 14), dtype=np.float64)
    state_estimates[:, 0]    = results["state_timestamps"]
    state_estimates[:, 1:4]  = results["positions"]
    state_estimates[:, 4:7]  = results["velocities"]
    state_estimates[:, 7:11] = results["quaternions"]
    state_estimates[:, 11:]  = np.tile(bias, (N, 1))

    pos_var  = results["pos_var"]
    vel_var  = results["vel_var"]
    rot_var  = results["rot_var"]
    bias_var = np.asarray(results["bias_var"]).ravel()

    state0   = np.concatenate([pos_var[0], vel_var[0], rot_var[0], bias_var])
    rest     = np.concatenate([pos_var[1:], vel_var[1:], rot_var[1:]], axis=1)
    cov_diag = np.concatenate([state0, rest.ravel()]).astype(np.float64)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("state_estimates",                    data=state_estimates)
        f.create_dataset("gyro_bias",                          data=bias)
        f.create_dataset("residuals",                          data=results["residuals"].astype(np.float64))
        f.create_dataset("state_estimate_covariance_diagonal", data=cov_diag)
        f.create_dataset("uma_accelerations",                  data=results["uma_accelerations"].astype(np.float64))
        f.create_dataset("uma_variance",                       data=results["uma_var"].astype(np.float64))

    print(f"[optimizer] Results saved → {output_path}")
