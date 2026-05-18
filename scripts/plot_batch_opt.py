#!/usr/bin/env python3
"""
plot_batch_opt.py <results_dir> [--dataset <dataset_dir>]

Plots OD batch optimization results.

  results_dir   path to an od results folder, e.g. data/results/dataset_foo_1714000000000/
                od_result.json, state_estimates.csv, covariance.csv, residuals.csv
                are expected there.  landmark_measurements.csv is also read from here
                if present (copied there automatically by the OD pipeline).

  --dataset     override the dataset folder for simulation ground truth files
                (ground_truth_states.h5, orbit_measurements.h5); defaults to
                dataset_folder in od_result.json.  Comparison plots are skipped
                if those files are absent.

Plots are saved into <results_dir>/plots/.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pyqt

try:
    import spiceypy as spice
except ImportError:
    spice = None


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_od_results(results_dir: Path) -> dict:
    """Load state_estimates, covariance (may be None), residuals, and metadata."""
    results_dir = Path(results_dir)

    meta_path = results_dir / "od_result.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"od_result.json not found in {results_dir}")
    with open(meta_path) as f:
        meta = json.load(f)

    se_path = results_dir / "state_estimates.csv"
    if not se_path.exists():
        raise FileNotFoundError(f"state_estimates.csv not found in {results_dir}")
    state_estimates = np.genfromtxt(se_path, delimiter=",", skip_header=1)

    cov_path = results_dir / "covariance.csv"
    covariance = np.genfromtxt(cov_path, delimiter=",", skip_header=1) if cov_path.exists() else None

    dyn_res_path = results_dir / "dynamics_residuals.csv"
    dynamics_residuals = (np.genfromtxt(dyn_res_path, delimiter=",", skip_header=1)
                          if dyn_res_path.exists() else None)

    ldmk_res_path = results_dir / "landmark_residuals.csv"
    _lmk_raw = (np.genfromtxt(ldmk_res_path, delimiter=",", skip_header=1)
                if ldmk_res_path.exists() else None)
    if _lmk_raw is not None and _lmk_raw.ndim == 1 and _lmk_raw.size > 0:
        _lmk_raw = _lmk_raw.reshape(1, -1)
    if _lmk_raw is not None and _lmk_raw.ndim == 2 and _lmk_raw.shape[1] >= 4:
        landmark_residuals = _lmk_raw[:, :3]
        landmark_outlier_flags = _lmk_raw[:, 3].astype(bool)
    else:
        landmark_residuals = _lmk_raw
        landmark_outlier_flags = (np.zeros(len(_lmk_raw), dtype=bool)
                                  if _lmk_raw is not None else None)

    return {
        "meta": meta,
        "state_estimates":       state_estimates,       # Nx11 array
        "covariance":            covariance,            # Nx10 array (timestamp+pos+vel+rot) or None
        "dynamics_residuals":    dynamics_residuals,    # (N-1)x13 array or None
        "landmark_residuals":    landmark_residuals,    # Mx3 array or None
        "landmark_outlier_flags": landmark_outlier_flags,  # (M,) bool or None
    }


def load_h5(path: Path) -> dict:
    """Load all datasets from an HDF5 file into a dict {dataset_path: ndarray}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    data = {}
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                data[name] = obj[()]
        f.visititems(visitor)
    return data


J2000_EPOCH_UNIX_S = 946727936.0


def measurement_search_dirs(results_dir: Path, dataset_dir: Path):
    dirs = []
    for base in (results_dir, dataset_dir, Path(dataset_dir).parent):
        if base and base not in dirs:
            dirs.append(Path(base))
    return dirs


def load_landmark_timestamps(results_dir: Path, dataset_dir: Path):
    """Return landmark timestamps in J2000 seconds from landmark_measurements.csv, or None.

    Checks results_dir first (copied there by the OD pipeline), then falls back
    to dataset_dir.
    """
    for base in measurement_search_dirs(results_dir, dataset_dir):
        csv = Path(base) / "landmark_measurements.csv"
        if csv.exists():
            data = np.genfromtxt(csv, delimiter=",", skip_header=1)
            if data.ndim == 1 and data.size > 0:
                return np.array([data[0] / 1000.0 - J2000_EPOCH_UNIX_S])
            if data.ndim >= 2:
                return data[:, 0] / 1000.0 - J2000_EPOCH_UNIX_S
    return None


def load_imu_timestamps(results_dir: Path, dataset_dir: Path):
    """Return IMU timestamps in J2000 seconds from imu_data.csv, or None."""
    for base in measurement_search_dirs(results_dir, dataset_dir):
        csv = Path(base) / "imu_data.csv"
        if csv.exists():
            data = np.genfromtxt(csv, delimiter=",", skip_header=1)
            if data.ndim == 1 and data.size > 0:
                return np.array([data[0] / 1000.0 - J2000_EPOCH_UNIX_S])
            if data.ndim >= 2:
                return data[:, 0] / 1000.0 - J2000_EPOCH_UNIX_S
    return None


# ── Processing helpers ─────────────────────────────────────────────────────────

def process_residuals(dynamics_residuals, landmark_residuals, bias_mode):
    # dynamics_residuals columns (StateResIdx): 0=timestamp, 1:4=pos, 4:7=vel, 7:10=rot, 10:13=bias
    lindynres   = dynamics_residuals[:, 1:7]    # pos + vel
    angdynres   = (dynamics_residuals[:, 7:13]  # rot + bias (TV_BIAS)
                   if bias_mode == "tv_bias"
                   else dynamics_residuals[:, 7:10])  # rot only
    ldmkmeasres = landmark_residuals[:, :3]
    return lindynres, angdynres, ldmkmeasres


# ── Frame conversion helpers ───────────────────────────────────────────────────

SPICE_KERNELS_LOADED = False
EARTH_RATE_RAD_S = 7.2921150e-5


def load_spice_kernels(kernel_dir: Path):
    """Load the kernels needed for J2000 <-> ITRF93 transforms."""
    global SPICE_KERNELS_LOADED
    if spice is None:
        raise RuntimeError("spiceypy is not installed; install it to generate ECEF plots.")
    if SPICE_KERNELS_LOADED:
        return

    kernel_dir = Path(kernel_dir)
    for name in ("naif0012.tls", "pck00011.tpc", "earth_latest_high_prec.bpc", "de440.bsp"):
        path = kernel_dir / name
        if not path.exists():
            raise FileNotFoundError(f"SPICE kernel not found: {path}")
        spice.furnsh(str(path))
    SPICE_KERNELS_LOADED = True


def unix_to_spice_et(unix_s):
    """Convert Unix UTC seconds to SPICE ET using the loaded leap-second kernel."""
    arr = np.atleast_1d(np.asarray(unix_s, dtype=float))
    et = np.array([
        spice.utc2et(datetime.fromtimestamp(float(t), timezone.utc)
                    .strftime("%Y-%m-%dT%H:%M:%S.%f"))
        for t in arr
    ])
    return et if np.ndim(unix_s) else et[0]


def _quat_wxyz_to_matrix(q_wxyz):
    return pyqt.Quaternion(*q_wxyz).normalised.rotation_matrix


def _matrix_to_quat_wxyz(matrix):
    q = pyqt.Quaternion(matrix=matrix).normalised
    out = np.array([q.w, q.x, q.y, q.z])
    return out * np.sign(out[0] if out[0] != 0 else 1.0)


def transform_estimates_eci_to_ecef(est_states, kernel_dir: Path):
    """Return a copy of state_estimates with r/v/q converted from ECI to ECEF.

    Velocity is Earth-relative rotating-frame velocity via SPICE sxform.
    The quaternion remains body-to-frame, now body-to-ECEF.
    """
    load_spice_kernels(kernel_dir)
    out = np.array(est_states, copy=True)
    unix_t = out[:, 0] + J2000_EPOCH_UNIX_S
    et = unix_to_spice_et(unix_t)

    for i, t_et in enumerate(et):
        xform = np.asarray(spice.sxform("J2000", "ITRF93", float(t_et)))
        state_ecef = xform @ out[i, 1:7]
        out[i, 1:7] = state_ecef

        r_eci_to_ecef = xform[:3, :3]
        q_body_to_eci = out[i, [10, 7, 8, 9]]
        r_body_to_ecef = r_eci_to_ecef @ _quat_wxyz_to_matrix(q_body_to_eci)
        q_body_to_ecef = _matrix_to_quat_wxyz(r_body_to_ecef)
        out[i, 7:11] = q_body_to_ecef[[1, 2, 3, 0]]

    return out


def transform_truth_eci_to_ecef(true_states, kernel_dir: Path):
    """Return a copy of ground truth states converted from ECI to ECEF."""
    load_spice_kernels(kernel_dir)
    out = {k: np.array(v, copy=True) for k, v in true_states.items()}
    unix_t = np.asarray(out["unixtime"], dtype=float)
    et = unix_to_spice_et(unix_t)
    omega_earth_ecef = np.array([0.0, 0.0, EARTH_RATE_RAD_S])

    for i, t_et in enumerate(et):
        xform = np.asarray(spice.sxform("J2000", "ITRF93", float(t_et)))
        out["states"][i, :6] = xform @ out["states"][i, :6]

        r_eci_to_ecef = xform[:3, :3]
        q_body_to_eci = out["states"][i, 6:10]
        r_body_to_ecef = r_eci_to_ecef @ _quat_wxyz_to_matrix(q_body_to_eci)
        out["states"][i, 6:10] = _matrix_to_quat_wxyz(r_body_to_ecef)

        # Angular rate stays expressed in body axes.  In the ECEF state, use
        # body wrt the rotating Earth frame: w_B/E = w_B/I - w_E/I.
        earth_rate_body = r_body_to_ecef.T @ omega_earth_ecef
        out["states"][i, 10:13] = out["states"][i, 10:13] - earth_rate_body

    return out


def transform_covariance_eci_to_ecef(covariance, est_states_eci, kernel_dir: Path):
    """Approximate diagonal ECEF covariances from diagonal ECI covariances."""
    if covariance is None:
        return None

    load_spice_kernels(kernel_dir)
    out = np.array(covariance, copy=True)
    unix_t = est_states_eci[:, 0] + J2000_EPOCH_UNIX_S
    et = unix_to_spice_et(unix_t)

    for i, t_et in enumerate(et):
        xform = np.asarray(spice.sxform("J2000", "ITRF93", float(t_et)))
        cart_cov_eci = np.diag(covariance[i, 1:7])
        cart_cov_ecef = xform @ cart_cov_eci @ xform.T
        out[i, 1:4] = np.diag(cart_cov_ecef[:3, :3])
        out[i, 4:7] = np.diag(cart_cov_ecef[3:, 3:])
        out[i, 7:10] = covariance[i, 7:10]

    return np.maximum(out, 0.0)


def _earth_rate_body_from_state(est_states):
    omega_earth_ecef = np.array([0.0, 0.0, EARTH_RATE_RAD_S])
    earth_rate_body = np.zeros((est_states.shape[0], 3))
    for i in range(est_states.shape[0]):
        q_body_to_ecef = est_states[i, [10, 7, 8, 9]]
        earth_rate_body[i] = _quat_wxyz_to_matrix(q_body_to_ecef).T @ omega_earth_ecef
    return earth_rate_body


def nearest_time_errors(meas_t_j2000, state_t_j2000):
    """Signed error from each measurement timestamp to the nearest state timestamp."""
    if meas_t_j2000 is None or len(meas_t_j2000) == 0:
        return None
    state_t = np.asarray(state_t_j2000, dtype=float)
    meas_t = np.asarray(meas_t_j2000, dtype=float)
    idx = np.searchsorted(state_t, meas_t)
    idx_lo = np.clip(idx - 1, 0, len(state_t) - 1)
    idx_hi = np.clip(idx, 0, len(state_t) - 1)
    nearest = np.where(np.abs(meas_t - state_t[idx_lo]) <= np.abs(meas_t - state_t[idx_hi]),
                       state_t[idx_lo], state_t[idx_hi])
    return meas_t - nearest


def _sqrt_nonnegative(values):
    return np.sqrt(np.maximum(values, 0.0))


# ── State plots ────────────────────────────────────────────────────────────────

def plot_states(est_states, out_dir, true_states=None, orbit_measurements=None,
                bias_fixed=None, frame_label="ECI", filename_prefix="eci"):
    """Plot state estimates; overlay ground truth when true_states is provided."""
    have_gt = true_states is not None
    colors  = ["C0", "C1"]

    if have_gt:
        t0     = true_states["unixtime"][0] if len(true_states["unixtime"]) > 0 else 0.0
        t_true = true_states["unixtime"] - t0
        t_est  = (est_states[:, 0] + J2000_EPOCH_UNIX_S) - t0
    else:
        t_est  = est_states[:, 0] - est_states[0, 0]

    def _save(fig, name):
        plt.tight_layout(); plt.subplots_adjust(top=0.92)
        plt.savefig(out_dir / name); plt.close()

    # Position
    est_pos = est_states[:, 1:4]
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        if have_gt:
            ax.plot(t_true, true_states["states"][:, i],  label="true", color=colors[0])
            ax.plot(t_est,  est_pos[:, i], label="est",  color=colors[1], linestyle="--")
        else:
            ax.plot(t_est, est_pos[:, i], color=colors[0])
        ax.set_ylabel(["Pos X (km)", "Pos Y (km)", "Pos Z (km)"][i]); ax.grid(True)
        if i == 0 and have_gt: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle(f"{frame_label} Position: True vs Estimated" if have_gt
                 else f"{frame_label} Position Estimate")
    _save(fig, f"{filename_prefix}_position.png")

    # Velocity
    est_vel = est_states[:, 4:7]
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        if have_gt:
            ax.plot(t_true, true_states["states"][:, 3 + i], label="true", color=colors[0])
            ax.plot(t_est,  est_vel[:, i], label="est",  color=colors[1], linestyle="--")
        else:
            ax.plot(t_est, est_vel[:, i], color=colors[0])
        ax.set_ylabel(["Vel X (km/s)", "Vel Y (km/s)", "Vel Z (km/s)"][i]); ax.grid(True)
        if i == 0 and have_gt: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle(f"{frame_label} Velocity: True vs Estimated" if have_gt
                 else f"{frame_label} Velocity Estimate")
    _save(fig, f"{filename_prefix}_velocity.png")

    # Quaternion — est CSV order: quat_x, quat_y, quat_z, quat_w (cols 7–10)
    # Reorder to [w, x, y, z] for display; flip sign for consistent hemisphere
    est_quat = est_states[:, [10, 7, 8, 9]]
    est_quat = est_quat * np.sign(est_quat[:, 0:1])
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        if have_gt:
            true_quat = true_states["states"][:, 6:10]
            true_quat = true_quat * np.sign(true_quat[:, 0:1])
            ax.plot(t_true, true_quat[:, i], label="true", color=colors[0])
            ax.plot(t_est,  est_quat[:, i],  label="est",  color=colors[1], linestyle="--")
        else:
            ax.plot(t_est, est_quat[:, i], color=colors[0])
        ax.set_ylabel(["QW", "QX", "QY", "QZ"][i]); ax.grid(True)
        if i == 0 and have_gt: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle(f"Body-to-{frame_label} Quaternion: True vs Estimated" if have_gt
                 else f"Body-to-{frame_label} Quaternion Estimate")
    _save(fig, f"{filename_prefix}_quaternion.png")

    # Gyro bias
    est_bias = (np.tile(bias_fixed, (len(t_est), 1)) if bias_fixed is not None
                else est_states[:, 11:14])
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        if have_gt:
            true_bias = true_states["states"][:, 13:16]
            ax.plot(t_true, true_bias[:, i], label="true", color=colors[0])
            ax.plot(t_est,  est_bias[:, i],  label="est",  color=colors[1], linestyle="--")
        else:
            ax.plot(t_est, est_bias[:, i], color=colors[0])
        ax.set_ylabel(["Bias X (rad/s)", "Bias Y (rad/s)", "Bias Z (rad/s)"][i]); ax.grid(True)
        if i == 0 and have_gt: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("Gyro Bias: True vs Estimated" if have_gt else "Gyro Bias Estimate")
    _save(fig, "gyro_bias.png")

    # Angular velocity (GT only — requires gyro measurements)
    if have_gt and orbit_measurements is not None:
        true_ang_vel = true_states["states"][:, 10:13]
        gyro         = orbit_measurements["gyro_measurements"]
        gyro_t       = gyro[:, 0]
        gyro_t_rel   = (gyro_t - t0 if np.nanmedian(gyro_t) > 1e9
                        else (gyro_t + J2000_EPOCH_UNIX_S) - t0)
        gyro_meas    = np.zeros((len(t_est), 3))
        for i in range(3):
            gyro_meas[:, i] = np.interp(t_est, gyro_t_rel, gyro[:, i + 1])
        est_ang_vel  = gyro_meas - est_bias
        if frame_label == "ECEF":
            est_ang_vel = est_ang_vel - _earth_rate_body_from_state(est_states)
        fig, axs     = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
        for i, ax in enumerate(axs):
            ax.plot(t_true, true_ang_vel[:, i], label="true", color=colors[0])
            ax.plot(t_est,  est_ang_vel[:, i],  label="est",  color=colors[1], linestyle="--")
            ax.set_ylabel(["Omega X (rad/s)", "Omega Y (rad/s)", "Omega Z (rad/s)"][i])
            ax.grid(True)
            if i == 0: ax.legend(loc="upper right")
        axs[-1].set_xlabel("time (s) since start")
        ref_frame = "ECEF" if frame_label == "ECEF" else "ECI"
        fig.suptitle(f"Body Angular Rate wrt {ref_frame} Expressed in Body Frame: True vs Estimated")
        _save(fig, f"{filename_prefix}_angular_velocity.png")


def plot_residuals_standalone(dynamics_residuals, landmark_residuals, bias_mode,
                              ldmk_t, out_dir, landmark_outlier_flags=None):
    t_dyn = dynamics_residuals[:, 0] - dynamics_residuals[0, 0]

    # Linear dynamics residuals (pos + vel)
    lindynres = dynamics_residuals[:, 1:7]
    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(10, 8))
    labels = ["Pos X", "Pos Y", "Pos Z", "Vel X", "Vel Y", "Vel Z"]
    for i, ax in enumerate(axs):
        ax.plot(t_dyn, lindynres[:, i], color="C0")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_ylabel(labels[i]); ax.grid(True)
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("Linear Dynamics Residuals")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "linear_dynamics_residuals.png"); plt.close()

    # Angular dynamics residuals
    if bias_mode == "tv_bias":
        angdynres = dynamics_residuals[:, 7:13]
        ang_labels = ["Rot X", "Rot Y", "Rot Z", "Bias X", "Bias Y", "Bias Z"]
    else:
        angdynres = dynamics_residuals[:, 7:10]
        ang_labels = ["Rot X", "Rot Y", "Rot Z"]
    fig, axs = plt.subplots(len(ang_labels), 1, sharex=True, figsize=(10, 6))
    axs = np.atleast_1d(axs)
    for i, ax in enumerate(axs):
        ax.plot(t_dyn, angdynres[:, i], color="C0")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_ylabel(ang_labels[i]); ax.grid(True)
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("Angular Dynamics Residuals")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "angular_dynamics_residuals.png"); plt.close()

    # Landmark measurement residuals
    if ldmk_t is not None:
        t_ldmk = ldmk_t - ldmk_t[0]
        xlabel = "time (s) since start"
    else:
        t_ldmk = np.arange(landmark_residuals.shape[0])
        xlabel = "measurement index"
    is_outlier = (landmark_outlier_flags
                  if landmark_outlier_flags is not None
                  else np.zeros(len(t_ldmk), dtype=bool))
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        ax.plot(t_ldmk[~is_outlier], landmark_residuals[~is_outlier, i],
                linestyle="None", marker=".", color="C0", label="inlier")
        if is_outlier.any():
            ax.plot(t_ldmk[is_outlier], landmark_residuals[is_outlier, i],
                    linestyle="None", marker=".", color="red", label="outlier")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_ylabel(["Res X", "Res Y", "Res Z"][i]); ax.grid(True)
        if i == 0 and is_outlier.any():
            ax.legend(loc="upper right")
    axs[-1].set_xlabel(xlabel)
    fig.suptitle("Landmark Measurement Residuals")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "landmark_residuals.png"); plt.close()


# ── Comparison plots (ground truth required) ───────────────────────────────────

def plot_errors(true_states, est_states, est_covars, bias_mode, out_dir,
                bias_fixed=None, bias_cov_fixed=None, frame_label="ECI",
                filename_prefix="eci", orbit_measurements=None):
    true_time = true_states["unixtime"]                                # Unix seconds
    t0        = true_time[0] if len(true_time) > 0 else 0
    t_true    = true_time - t0
    t_est     = (est_states[:, 0] + J2000_EPOCH_UNIX_S) - t0          # J2000 → Unix → relative
    colors    = ["C0", "C1"]
    have_cov  = est_covars is not None

    # Position error
    true_pos           = true_states["states"][:, :3]
    est_pos            = est_states[:, 1:4]
    true_pos_at_est    = np.zeros(est_pos.shape)
    est_covars_pos     = _sqrt_nonnegative(est_covars[:, 1:4]) if have_cov else None
    for i in range(3):
        true_pos_at_est[:, i] = np.interp(t_est, t_true, true_pos[:, i])
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        ax.plot(t_est, true_pos_at_est[:, i] - est_pos[:, i], label="error", color=colors[0])
        if have_cov:
            ax.fill_between(t_est, -3 * est_covars_pos[:, i], 3 * est_covars_pos[:, i],
                            color=colors[0], alpha=0.3, label="3-sigma")
        ax.set_ylabel(["Error X (km)", "Error Y (km)", "Error Z (km)"][i])
        ax.grid(True)
        if i == 0: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle(f"{frame_label} Position: Error Three Axis")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / f"{filename_prefix}_position_three_error.png")
    plt.close()

    pos_norm           = np.linalg.norm(true_pos_at_est - est_pos, axis=1)
    est_covars_pos_norm = _sqrt_nonnegative(est_covars[:, 1:4].sum(axis=1)) if have_cov else None
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_est, pos_norm, label="Position Error Norm", color="C0")
    if have_cov:
        ax.fill_between(t_est, np.zeros_like(t_est), 3 * est_covars_pos_norm,
                        color="C0", alpha=0.3, label="3-sigma")
    ax.set_ylabel("Error Norm (km)"); ax.set_xlabel("time (s) since start")
    ax.set_title(f"{frame_label} Position Error Norm"); ax.grid(True); ax.legend(loc="upper right")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / f"{filename_prefix}_position_error_norm.png")
    plt.close()

    # Velocity error
    true_vel        = true_states["states"][:, 3:6]
    est_vel         = est_states[:, 4:7]
    true_vel_at_est = np.zeros(est_vel.shape)
    vel_covar       = _sqrt_nonnegative(est_covars[:, 4:7]) if have_cov else None
    for i in range(3):
        true_vel_at_est[:, i] = np.interp(t_est, t_true, true_vel[:, i])
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        ax.plot(t_est, true_vel_at_est[:, i] - est_vel[:, i], label="error", color=colors[0])
        if have_cov:
            ax.fill_between(t_est, -3 * vel_covar[:, i], 3 * vel_covar[:, i],
                            color=colors[0], alpha=0.3, label="3-sigma")
        ax.set_ylabel(["Error X (km/s)", "Error Y (km/s)", "Error Z (km/s)"][i])
        ax.grid(True)
        if i == 0: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle(f"{frame_label} Velocity: Error Three Axis")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / f"{filename_prefix}_velocity_three_error.png")
    plt.close()

    vel_norm            = np.linalg.norm(true_vel_at_est - est_vel, axis=1)
    est_covars_vel_norm = _sqrt_nonnegative(est_covars[:, 4:7].sum(axis=1)) if have_cov else None
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_est, vel_norm, label="Velocity Error Norm", color="C0")
    if have_cov:
        ax.fill_between(t_est, np.zeros_like(t_est), 3 * est_covars_vel_norm,
                        color="C0", alpha=0.3, label="3-sigma")
    ax.set_ylabel("Error Norm (km/s)"); ax.set_xlabel("time (s) since start")
    ax.set_title(f"{frame_label} Velocity Error Norm"); ax.grid(True); ax.legend(loc="upper right")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / f"{filename_prefix}_velocity_error_norm.png")
    plt.close()

    # Attitude error
    true_quat     = true_states["states"][:, 6:10]
    true_quat     = true_quat * np.sign(true_quat[:, 0:1])
    est_quat      = est_states[:, [10, 7, 8, 9]]
    est_quat      = est_quat * np.sign(est_quat[:, 0:1])
    true_quat_at_est = np.zeros(est_quat.shape)
    for i in range(4):
        true_quat_at_est[:, i] = np.interp(t_est, t_true, true_quat[:, i])

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        ax.plot(t_est, true_quat_at_est[:, i] - est_quat[:, i], label="error", color=colors[0])
        ax.set_ylabel(["QW", "QX", "QY", "QZ"][i]); ax.grid(True)
        if i == 0: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle(f"Body-to-{frame_label} Quaternion: Error Four Axis")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / f"{filename_prefix}_quaternion_four_error.png")
    plt.close()

    att_covar        = np.rad2deg(_sqrt_nonnegative(est_covars[:, 7:10])) if have_cov else None
    angle_errors     = np.zeros((est_quat.shape[0], 3))
    angle_error_norm = np.zeros(est_quat.shape[0])
    att_norm_std     = np.linalg.norm(att_covar, axis=1) if have_cov else None
    for i in range(est_quat.shape[0]):
        q_true = pyqt.Quaternion(*true_quat_at_est[i])
        q_est  = pyqt.Quaternion(*est_quat[i])
        dq     = q_true.inverse * q_est
        angle_errors[i]     = np.rad2deg(dq.axis * dq.angle)
        angle_error_norm[i] = np.rad2deg(2 * np.arccos(np.clip(np.abs(dq.w), -1.0, 1.0)))

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 4))
    for i, ax in enumerate(axs):
        ax.plot(t_est, angle_errors[:, i], label="error", color=colors[0])
        if have_cov:
            ax.fill_between(t_est, -3 * att_covar[:, i], 3 * att_covar[:, i],
                            color=colors[0], alpha=0.3, label="3-sigma")
        ax.set_ylabel(["X (deg)", "Y (deg)", "Z (deg)"][i]); ax.grid(True)
        if i == 0: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle(f"Body-to-{frame_label} Attitude Error")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / f"{filename_prefix}_attitude_error.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_est, angle_error_norm, label="Attitude Error", color="C0")
    if have_cov:
        ax.fill_between(t_est, np.zeros_like(t_est), 3 * att_norm_std,
                        color="C0", alpha=0.3, label="3-sigma")
    ax.set_ylabel("Error (degrees)"); ax.set_xlabel("time (s) since start")
    ax.set_title(f"Body-to-{frame_label} Attitude Error Norm"); ax.grid(True)
    ylim_ref = max(np.max(angle_error_norm), np.max(3 * att_norm_std)) if have_cov else np.max(angle_error_norm)
    ax.set_ylim(0, np.minimum(ylim_ref * 1.1, 180))
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / f"{filename_prefix}_attitude_error_norm.png")
    plt.close()

    est_bias  = (np.tile(bias_fixed, (len(t_est), 1)) if bias_fixed is not None
                 else est_states[:, 11:14])

    # Angular-rate error. Rates remain expressed in body axes; the frame label
    # indicates whether the rate is body wrt inertial or body wrt rotating ECEF.
    if orbit_measurements is not None:
        true_ang_vel = true_states["states"][:, 10:13]
        true_ang_vel_at_est = np.zeros((len(t_est), 3))
        for i in range(3):
            true_ang_vel_at_est[:, i] = np.interp(t_est, t_true, true_ang_vel[:, i])

        gyro = orbit_measurements["gyro_measurements"]
        gyro_t = gyro[:, 0]
        gyro_t_rel = (gyro_t - t0 if np.nanmedian(gyro_t) > 1e9
                      else (gyro_t + J2000_EPOCH_UNIX_S) - t0)
        gyro_at_est = np.zeros((len(t_est), 3))
        for i in range(3):
            gyro_at_est[:, i] = np.interp(t_est, gyro_t_rel, gyro[:, i + 1])

        est_ang_vel = gyro_at_est - est_bias
        if frame_label == "ECEF":
            est_ang_vel = est_ang_vel - _earth_rate_body_from_state(est_states)
        ang_vel_error = true_ang_vel_at_est - est_ang_vel

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
        for i, ax in enumerate(axs):
            ax.plot(t_est, ang_vel_error[:, i], label="error", color=colors[0])
            ax.axhline(0, color="k", linewidth=0.5)
            ax.set_ylabel(["Omega X (rad/s)", "Omega Y (rad/s)", "Omega Z (rad/s)"][i])
            ax.grid(True)
            if i == 0: ax.legend(loc="upper right")
        axs[-1].set_xlabel("time (s) since start")
        ref_frame = "ECEF" if frame_label == "ECEF" else "ECI"
        fig.suptitle(f"Body Angular Rate wrt {ref_frame} Error (Body Frame)")
        plt.tight_layout(); plt.subplots_adjust(top=0.92)
        plt.savefig(out_dir / f"{filename_prefix}_angular_velocity_three_error.png")
        plt.close()

        ang_vel_norm = np.linalg.norm(ang_vel_error, axis=1)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_est, ang_vel_norm, label="Angular Rate Error Norm", color="C0")
        ax.set_ylabel("Error Norm (rad/s)"); ax.set_xlabel("time (s) since start")
        ax.set_title(f"Body Angular Rate wrt {ref_frame} Error Norm")
        ax.grid(True); ax.legend(loc="upper right")
        plt.tight_layout(); plt.subplots_adjust(top=0.92)
        plt.savefig(out_dir / f"{filename_prefix}_angular_velocity_error_norm.png")
        plt.close()

    # Gyro bias error
    true_bias = true_states["states"][:, 13:16]
    true_bias_at_est = np.zeros(est_bias.shape)
    for i in range(3):
        true_bias_at_est[:, i] = np.interp(t_est, t_true, true_bias[:, i])
    if not have_cov or bias_mode == "no_bias":
        gyro_bias_covar = np.zeros(est_bias.shape)
    elif bias_mode == "fix_bias":
        if bias_cov_fixed is not None:
            gyro_bias_covar = np.tile(_sqrt_nonnegative(np.asarray(bias_cov_fixed)), (len(t_est), 1))
        else:
            gyro_bias_covar = np.zeros((len(t_est), 3))
    else:
        gyro_bias_covar = _sqrt_nonnegative(est_covars[:, 10:13])
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        ax.plot(t_est, true_bias_at_est[:, i] - est_bias[:, i], label="error", color=colors[0])
        if have_cov:
            ax.fill_between(t_est, -3 * gyro_bias_covar[:, i], 3 * gyro_bias_covar[:, i],
                            color=colors[0], alpha=0.3, label="3-sigma")
        ax.set_ylabel(["Bias X (rad/s)", "Bias Y (rad/s)", "Bias Z (rad/s)"][i]); ax.grid(True)
        if i == 0: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("Gyro Bias: Error Three Axis")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "gyro_bias_three_error.png")
    plt.close()

    gyro_bias_norm           = np.linalg.norm(true_bias_at_est - est_bias, axis=1)
    est_covars_gyro_bias_norm = np.linalg.norm(gyro_bias_covar, axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_est, gyro_bias_norm, label="Gyro Bias Error Norm", color="C0")
    if have_cov:
        ax.fill_between(t_est, np.zeros_like(t_est), 3 * est_covars_gyro_bias_norm,
                        color="C0", alpha=0.3, label="3-sigma")
    ax.set_ylabel("Error Norm (rad/s)"); ax.set_xlabel("time (s) since start")
    ax.set_title("Gyro Bias Error Norm"); ax.grid(True); ax.legend(loc="upper right")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "gyro_bias_error_norm.png")
    plt.close()


def plot_measurements(measurements, ground_truth_states, est_states, ldmkmeasres, out_dir):
    true_time  = ground_truth_states["unixtime"]
    t0         = true_time[0] if len(true_time) > 0 else 0
    gyro_meas  = measurements["gyro_measurements"][:, 1:4]
    t_gyro     = measurements["gyro_measurements"][:, 0] - t0
    colors     = ["C0", "C1"]

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        ax.plot(t_gyro, gyro_meas[:, i], label="meas", color=colors[1], linestyle="--")
        ax.set_ylabel(["Omega X (rad/s)", "Omega Y (rad/s)", "Omega Z (rad/s)"][i]); ax.grid(True)
        if i == 0: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("Gyro measurements")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "gyro_measurements.png")
    plt.close()

    landmark_meas = measurements["landmark_measurements"]
    t_landmark    = landmark_meas[:, 0] - t0
    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(10, 6))
    labels = ["Bear X", "Bear Y", "Bear Z", "Ldmk X (km)", "Ldmk Y (km)", "Ldmk Z (km)"]
    for i, ax in enumerate(axs):
        ax.plot(t_landmark, landmark_meas[:, i + 1], color=colors[1],
                linestyle="None", marker=".", label="meas")
        ax.set_ylabel(labels[i]); ax.grid(True)
        if i == 0: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("Landmark measurements")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "landmark_measurements.png")
    plt.close()


def plot_measurement_time_errors(est_states, ldmk_t, imu_t, out_dir):
    """Plot per-measurement timestamp offset to the nearest optimizer state."""
    state_t = est_states[:, 0]

    def _plot(meas_t, name, title):
        if meas_t is None:
            return
        errors = nearest_time_errors(meas_t, state_t)
        t_rel = meas_t - meas_t[0]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_rel, errors, linestyle="None", marker=".", color="C0")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("measurement time (s) since first measurement")
        ax.set_ylabel("measurement - nearest state (s)")
        ax.set_title(title)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / name)
        plt.close()

    _plot(ldmk_t, "landmark_time_errors.png", "Landmark Timestamp Error per Measurement")
    _plot(imu_t, "imu_time_errors.png", "IMU Timestamp Error per Measurement")


def plot_residuals(lindynres, angdynres, ldmkmeasres, ldmk_t, est_states,
                   true_states, bias_mode, out_dir, landmark_outlier_flags=None):
    true_time = true_states["unixtime"]                                # Unix seconds
    t0        = true_time[0] if len(true_time) > 0 else 0
    t_est     = (est_states[:, 0] + J2000_EPOCH_UNIX_S) - t0          # J2000 -> Unix -> relative
    colors    = ["C0", "C1"]

    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(10, 6))
    labels = ["LinDyn Res X", "LinDyn Res Y", "LinDyn Res Z",
              "LinDyn Res VX", "LinDyn Res VY", "LinDyn Res VZ"]
    for i, ax in enumerate(axs):
        ax.plot(t_est[:-1], lindynres[:, i], label="residual", color=colors[0])
        ax.fill_between(t_est[:-1], -3 * np.ones_like(t_est[:-1]), 3 * np.ones_like(t_est[:-1]),
                        color="C0", alpha=0.3, label="3-sigma")
        ax.set_ylabel(labels[i]); ax.grid(True)
        if i == 0: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time step")
    fig.suptitle("Linear Dynamics Residuals")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "linear_dynamics_residuals_gt.png")
    plt.close()

    n_angdyn = 3 if bias_mode in ["fix_bias", "no_bias"] else 6
    labels   = ["AngDyn Res X", "AngDyn Res Y", "AngDyn Res Z"]
    if n_angdyn == 6:
        labels += ["AngDyn Res Bias X", "AngDyn Res Bias Y", "AngDyn Res Bias Z"]
    fig, axs = plt.subplots(n_angdyn, 1, sharex=True, figsize=(10, 6))
    axs = np.atleast_1d(axs)
    for i, ax in enumerate(axs):
        ax.plot(t_est[:-1], angdynres[:, i], label="residual", color=colors[0])
        ax.fill_between(t_est[:-1], -3 * np.ones_like(t_est[:-1]), 3 * np.ones_like(t_est[:-1]),
                        color="C0", alpha=0.3, label="3-sigma")
        ax.set_ylabel(labels[i]); ax.grid(True)
        if i == 0: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("Angular Dynamics Residuals")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "angular_dynamics_residuals_gt.png")
    plt.close()

    if ldmk_t is not None:
        t_landmark = (ldmk_t + J2000_EPOCH_UNIX_S) - t0   # J2000 -> Unix -> relative
        xlabel_ldmk = "time (s) since start"
    else:
        t_landmark  = np.arange(ldmkmeasres.shape[0], dtype=float)
        xlabel_ldmk = "measurement index"
    is_outlier = (landmark_outlier_flags
                  if landmark_outlier_flags is not None
                  else np.zeros(len(t_landmark), dtype=bool))
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        ax.plot(t_landmark[~is_outlier], ldmkmeasres[~is_outlier, i],
                linestyle="None", marker=".", color=colors[0], label="inlier")
        if is_outlier.any():
            ax.plot(t_landmark[is_outlier], ldmkmeasres[is_outlier, i],
                    linestyle="None", marker=".", color="red", label="outlier")
        ax.fill_between(t_landmark, -3, 3, color="C0", alpha=0.3, label="3-sigma")
        ax.set_ylabel(["Ldmk Meas Res X", "Ldmk Meas Res Y", "Ldmk Meas Res Z"][i]); ax.grid(True)
        if i == 0: ax.legend(loc="upper right")
    axs[-1].set_xlabel(xlabel_ldmk)
    fig.suptitle("Landmark Measurement Residuals")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "landmark_measurement_residuals_gt.png")
    plt.close()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path,
                        help="OD results folder (contains od_result.json, state_estimates.csv, …)")
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Dataset folder with ground_truth_states.h5 and orbit_measurements.h5; "
                             "defaults to dataset_folder in od_result.json")
    parser.add_argument("--kernel-dir", type=Path, default=Path("data/kernels"),
                        help="Directory containing SPICE kernels for ECEF post-processing")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load OD results
    od = load_od_results(results_dir)
    meta        = od["meta"]
    est_states  = od["state_estimates"]
    covariance         = od["covariance"]
    dynamics_residuals = od["dynamics_residuals"]
    landmark_residuals     = od["landmark_residuals"]
    landmark_outlier_flags = od["landmark_outlier_flags"]
    bias_mode              = meta.get("bias_mode", "fix_bias")
    if isinstance(bias_mode, int):
        bias_mode = {0: "no_bias", 1: "fix_bias", 2: "tv_bias"}.get(bias_mode, "fix_bias")

    print(f"Loaded OD results from {results_dir}")
    print(f"  error_code : {meta.get('error_code')}")
    print(f"  bias_mode  : {bias_mode}")
    solver_info = meta.get("solver", {})
    print(f"  solver     : {solver_info.get('return_status')} "
          f"({solver_info.get('iter_count')} iters, "
          f"cost {solver_info.get('final_cost', 0.0):.3f})")
    print(f"  states     : {est_states.shape}")

    estimates = meta.get("estimates", {})
    bias_fixed = None
    bias_cov_fixed = None
    if "gyro_bias_x_rads" in estimates:
        bias_fixed = np.array([
            estimates["gyro_bias_x_rads"],
            estimates["gyro_bias_y_rads"],
            estimates["gyro_bias_z_rads"],
        ])
        sigma_bias = (np.sqrt(np.maximum(np.array([
            estimates["gyro_bias_cov_x_rads2"],
            estimates["gyro_bias_cov_y_rads2"],
            estimates["gyro_bias_cov_z_rads2"],
        ]), 0.0)) if "gyro_bias_cov_x_rads2" in estimates else None)
        bias_cov_fixed = (np.array([
            estimates["gyro_bias_cov_x_rads2"],
            estimates["gyro_bias_cov_y_rads2"],
            estimates["gyro_bias_cov_z_rads2"],
        ]) if "gyro_bias_cov_x_rads2" in estimates else None)
        for i, axis in enumerate("xyz"):
            if sigma_bias is not None:
                print(f"  gyro_bias_{axis} : {bias_fixed[i]:.6e} ± {sigma_bias[i]:.2e} rad/s")
            else:
                print(f"  gyro_bias_{axis} : {bias_fixed[i]:.6e} rad/s")
    cd = estimates.get("cd")
    if cd is not None:
        cd_var = estimates.get("cd_var")
        if cd_var is not None:
            print(f"  Cd         : {cd:.4e} ± {cd_var**0.5:.2e}")
        else:
            print(f"  Cd         : {cd:.4e}")

    # Resolve dataset folder
    dataset_dir = args.dataset or Path(meta.get("dataset_folder", ""))

    # Output folder for plots
    out_dir = results_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    print(f"Saving plots to {out_dir}")

    # Load ground truth and measurements if available (simulation only)
    gt_path   = dataset_dir / "ground_truth_states.h5"
    meas_path = dataset_dir / "orbit_measurements.h5"
    have_gt   = gt_path.exists() and meas_path.exists()

    if have_gt:
        ground_truth  = load_h5(gt_path)
        measurements  = load_h5(meas_path)
        print(f"Loaded ground truth from {gt_path}")
        print(f"Loaded measurements from {meas_path}")
    else:
        print("Ground truth / measurements not found — skipping comparison plots.")

    # Residual decomposition
    if dynamics_residuals is not None and landmark_residuals is not None:
        lindynres, angdynres, ldmkmeasres = process_residuals(
            dynamics_residuals, landmark_residuals, bias_mode)
    else:
        lindynres = angdynres = ldmkmeasres = None

    # Covariance is Nx10 (timestamp + pos + vel + rot; gyro_bias_cov in od_result.json)
    est_covars = covariance

    ldmk_t = load_landmark_timestamps(results_dir, dataset_dir)
    imu_t = load_imu_timestamps(results_dir, dataset_dir)
    if imu_t is None and have_gt and "gyro_measurements" in measurements:
        gyro_t = measurements["gyro_measurements"][:, 0]
        imu_t = (gyro_t - J2000_EPOCH_UNIX_S if np.nanmedian(gyro_t) > 1e9
                 else gyro_t)

    # ── State estimate plots (always; overlaid with GT when available) ───────────
    plot_states(est_states, out_dir,
                true_states=ground_truth if have_gt else None,
                orbit_measurements=measurements if have_gt else None,
                bias_fixed=bias_fixed,
                frame_label="ECI",
                filename_prefix="eci")
    plot_measurement_time_errors(est_states, ldmk_t, imu_t, out_dir)

    # ECEF post-processing.  Position/velocity use SPICE's state transform so
    # ECEF velocity is relative to the rotating Earth frame.
    try:
        ecef_est_states = transform_estimates_eci_to_ecef(est_states, args.kernel_dir)
        ecef_ground_truth = (transform_truth_eci_to_ecef(ground_truth, args.kernel_dir)
                             if have_gt else None)
        ecef_covars = transform_covariance_eci_to_ecef(est_covars, est_states, args.kernel_dir)
        plot_states(ecef_est_states, out_dir,
                    true_states=ecef_ground_truth,
                    orbit_measurements=measurements if have_gt else None,
                    bias_fixed=bias_fixed,
                    frame_label="ECEF",
                    filename_prefix="ecef")
    except Exception as exc:
        print(f"Skipping ECEF plots: {exc}")
        ecef_est_states = ecef_ground_truth = ecef_covars = None

    if dynamics_residuals is not None and landmark_residuals is not None:
        plot_residuals_standalone(dynamics_residuals, landmark_residuals,
                                  bias_mode, ldmk_t, out_dir,
                                  landmark_outlier_flags=landmark_outlier_flags)

    # ── Comparison plots (simulation ground truth only) ───────────────────────
    if have_gt:
        plot_errors(ground_truth, est_states, est_covars, bias_mode, out_dir,
                    bias_fixed=bias_fixed, bias_cov_fixed=bias_cov_fixed,
                    frame_label="ECI", filename_prefix="eci",
                    orbit_measurements=measurements)
        if ecef_ground_truth is not None:
            plot_errors(ecef_ground_truth, ecef_est_states, ecef_covars, bias_mode, out_dir,
                        bias_fixed=bias_fixed, bias_cov_fixed=bias_cov_fixed,
                        frame_label="ECEF", filename_prefix="ecef",
                        orbit_measurements=measurements)
        plot_measurements(measurements, ground_truth, est_states, ldmkmeasres, out_dir)
        if ldmkmeasres is not None:
            plot_residuals(lindynres, angdynres, ldmkmeasres,
                           ldmk_t, est_states, ground_truth, bias_mode, out_dir,
                           landmark_outlier_flags=landmark_outlier_flags)

    print("Done.")
