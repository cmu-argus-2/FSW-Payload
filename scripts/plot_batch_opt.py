#!/usr/bin/env python3
"""
plot_batch_opt.py <results_dir> [--dataset <dataset_dir>]

Plots OD batch optimization results.

  results_dir   path to an od results folder, e.g. data/results/dataset_foo_1714000000000/
                od_result.json, state_estimates.csv, covariance.csv, residuals.csv
                are expected there.

  --dataset     override the dataset folder (default: read from od_result.json).
                Ground truth (ground_truth_states.h5) and measurements
                (orbit_measurements.h5) are looked for there; comparison plots
                are skipped if they are absent.

Plots are saved into <results_dir>/plots/.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pyquaternion as pyqt


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
    landmark_residuals = (np.genfromtxt(ldmk_res_path, delimiter=",", skip_header=1)
                          if ldmk_res_path.exists() else None)

    return {
        "meta": meta,
        "state_estimates":    state_estimates,    # Nx14 array
        "covariance":         covariance,         # Nx13 array (StateResIdx columns) or None
        "dynamics_residuals": dynamics_residuals, # (N-1)x13 array (StateResIdx columns) or None
        "landmark_residuals": landmark_residuals, # Mx3 array (LandmarkResIdx columns) or None
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


def load_landmark_timestamps(dataset_dir: Path):
    """Return landmark timestamps in J2000 seconds from landmark_measurements.csv, or None."""
    csv = Path(dataset_dir) / "landmark_measurements.csv"
    if not csv.exists():
        return None
    data = np.genfromtxt(csv, delimiter=",", skip_header=1)
    if data.ndim < 2:
        return None
    return data[:, 0] / 1000.0 - J2000_EPOCH_UNIX_S


# ── Processing helpers ─────────────────────────────────────────────────────────

def process_residuals(dynamics_residuals, landmark_residuals, bias_mode):
    # dynamics_residuals columns (StateResIdx): 0=timestamp, 1:4=pos, 4:7=vel, 7:10=rot, 10:13=bias
    lindynres   = dynamics_residuals[:, 1:7]    # pos + vel
    angdynres   = (dynamics_residuals[:, 7:13]  # rot + bias (TV_BIAS)
                   if bias_mode == "tv_bias"
                   else dynamics_residuals[:, 7:10])  # rot only
    ldmkmeasres = landmark_residuals[:, :3]
    return lindynres, angdynres, ldmkmeasres


# ── State plots ────────────────────────────────────────────────────────────────

def plot_states(est_states, out_dir, true_states=None, orbit_measurements=None):
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
    fig.suptitle("ECI Position: True vs Estimated" if have_gt else "ECI Position Estimate")
    _save(fig, "eci_position.png")

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
    fig.suptitle("ECI Velocity: True vs Estimated" if have_gt else "ECI Velocity Estimate")
    _save(fig, "eci_velocity.png")

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
    fig.suptitle("Quaternion: True vs Estimated" if have_gt else "Quaternion Estimate")
    _save(fig, "quaternion.png")

    # Gyro bias
    est_bias = est_states[:, 11:14]
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
        gyro_meas    = orbit_measurements["gyro_measurements"][:, 1:4]
        est_ang_vel  = gyro_meas - est_bias
        fig, axs     = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
        for i, ax in enumerate(axs):
            ax.plot(t_true, true_ang_vel[:, i], label="true", color=colors[0])
            ax.plot(t_est,  est_ang_vel[:, i],  label="est",  color=colors[1], linestyle="--")
            ax.set_ylabel(["Omega X (rad/s)", "Omega Y (rad/s)", "Omega Z (rad/s)"][i])
            ax.grid(True)
            if i == 0: ax.legend(loc="upper right")
        axs[-1].set_xlabel("time (s) since start")
        fig.suptitle("Angular Velocity: True vs Estimated")
        _save(fig, "angular_velocity.png")


def plot_residuals_standalone(dynamics_residuals, landmark_residuals, bias_mode,
                              ldmk_t, out_dir):
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
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        ax.plot(t_ldmk, landmark_residuals[:, i],
                linestyle="None", marker=".", color="C0")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_ylabel(["Res X", "Res Y", "Res Z"][i]); ax.grid(True)
    axs[-1].set_xlabel(xlabel)
    fig.suptitle("Landmark Measurement Residuals")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "landmark_residuals.png"); plt.close()


# ── Comparison plots (ground truth required) ───────────────────────────────────

def plot_errors(true_states, est_states, est_covars, bias_mode, out_dir):
    true_time = true_states["unixtime"]                                # Unix seconds
    t0        = true_time[0] if len(true_time) > 0 else 0
    t_true    = true_time - t0
    t_est     = (est_states[:, 0] + J2000_EPOCH_UNIX_S) - t0          # J2000 → Unix → relative
    colors    = ["C0", "C1"]

    # Position error
    true_pos           = true_states["states"][:, :3]
    est_pos            = est_states[:, 1:4]
    true_pos_at_est    = np.zeros(est_pos.shape)
    est_covars_pos     = np.sqrt(est_covars[:, 1:4])
    for i in range(3):
        true_pos_at_est[:, i] = np.interp(t_est, t_true, true_pos[:, i])
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        ax.plot(t_est, true_pos_at_est[:, i] - est_pos[:, i], label="error", color=colors[0])
        ax.fill_between(t_est, -3 * est_covars_pos[:, i], 3 * est_covars_pos[:, i],
                        color=colors[0], alpha=0.3, label="3-sigma")
        ax.set_ylabel(["Error X (km)", "Error Y (km)", "Error Z (km)"][i])
        ax.grid(True)
        if i == 0: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("ECI Position: Error Three Axis")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "eci_position_three_error.png")
    plt.close()

    pos_norm           = np.linalg.norm(true_pos_at_est - est_pos, axis=1)
    est_covars_pos_norm = np.sqrt(est_covars[:, 1:4].sum(axis=1))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_est, pos_norm, label="Position Error Norm", color="C0")
    ax.fill_between(t_est, np.zeros_like(t_est), 3 * est_covars_pos_norm,
                    color="C0", alpha=0.3, label="3-sigma")
    ax.set_ylabel("Error Norm (km)"); ax.set_xlabel("time (s) since start")
    ax.set_title("Position Error Norm"); ax.grid(True); ax.legend(loc="upper right")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "eci_position_error_norm.png")
    plt.close()

    # Velocity error
    true_vel        = true_states["states"][:, 3:6]
    est_vel         = est_states[:, 4:7]
    true_vel_at_est = np.zeros(est_vel.shape)
    vel_covar       = np.sqrt(est_covars[:, 4:7])
    for i in range(3):
        true_vel_at_est[:, i] = np.interp(t_est, t_true, true_vel[:, i])
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        ax.plot(t_est, true_vel_at_est[:, i] - est_vel[:, i], label="error", color=colors[0])
        ax.fill_between(t_est, -3 * vel_covar[:, i], 3 * vel_covar[:, i],
                        color=colors[0], alpha=0.3, label="3-sigma")
        ax.set_ylabel(["Error X (km/s)", "Error Y (km/s)", "Error Z (km/s)"][i])
        ax.grid(True)
        if i == 0: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("ECI Velocity: Error Three Axis")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "eci_velocity_three_error.png")
    plt.close()

    vel_norm            = np.linalg.norm(true_vel_at_est - est_vel, axis=1)
    est_covars_vel_norm = np.sqrt(est_covars[:, 4:7].sum(axis=1))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_est, vel_norm, label="Velocity Error Norm", color="C0")
    ax.fill_between(t_est, np.zeros_like(t_est), 3 * est_covars_vel_norm,
                    color="C0", alpha=0.3, label="3-sigma")
    ax.set_ylabel("Error Norm (km/s)"); ax.set_xlabel("time (s) since start")
    ax.set_title("Velocity Error Norm"); ax.grid(True); ax.legend(loc="upper right")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "eci_velocity_error_norm.png")
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
    fig.suptitle("Quaternion: Error Four Axis")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "quaternion_four_error.png")
    plt.close()

    att_covar        = np.rad2deg(np.sqrt(est_covars[:, 7:10]))
    angle_errors     = np.zeros((est_quat.shape[0], 3))
    angle_error_norm = np.zeros(est_quat.shape[0])
    att_norm_std     = np.linalg.norm(att_covar, axis=1)
    for i in range(est_quat.shape[0]):
        q_true = pyqt.Quaternion(*true_quat_at_est[i])
        q_est  = pyqt.Quaternion(*est_quat[i])
        dq     = q_true.inverse * q_est
        angle_errors[i]     = np.rad2deg(dq.axis * dq.angle)
        angle_error_norm[i] = np.rad2deg(2 * np.arccos(np.clip(np.abs(dq.w), -1.0, 1.0)))

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 4))
    for i, ax in enumerate(axs):
        ax.plot(t_est, angle_errors[:, i], label="error", color=colors[0])
        ax.fill_between(t_est, -3 * att_covar[:, i], 3 * att_covar[:, i],
                        color=colors[0], alpha=0.3, label="3-sigma")
        ax.set_ylabel(["X (deg)", "Y (deg)", "Z (deg)"][i]); ax.grid(True)
        if i == 0: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("Attitude Error")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "attitude_error.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_est, angle_error_norm, label="Attitude Error", color="C0")
    ax.fill_between(t_est, np.zeros_like(t_est), 3 * att_norm_std,
                    color="C0", alpha=0.3, label="3-sigma")
    ax.set_ylabel("Error (degrees)"); ax.set_xlabel("time (s) since start")
    ax.set_title("Attitude Error Norm"); ax.grid(True)
    ax.set_ylim(0, min(np.max(angle_error_norm) * 1.1, 180))
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "attitude_error_norm.png")
    plt.close()

    # Gyro bias error
    true_bias      = true_states["states"][:, 13:16]
    est_bias       = est_states[:, 11:14]
    true_bias_at_est = np.zeros(est_bias.shape)
    for i in range(3):
        true_bias_at_est[:, i] = np.interp(t_est, t_true, true_bias[:, i])
    gyro_bias_covar = (np.zeros(est_bias.shape) if bias_mode == "no_bias"
                       else np.sqrt(est_covars[:, 10:13]))
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        ax.plot(t_est, true_bias_at_est[:, i] - est_bias[:, i], label="error", color=colors[0])
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
    est_covars_gyro_bias_norm = np.sqrt(est_covars[:, 10:13].sum(axis=1))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_est, gyro_bias_norm, label="Gyro Bias Error Norm", color="C0")
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


def plot_residuals(lindynres, angdynres, ldmkmeasres, ldmk_t, est_states,
                   true_states, bias_mode, out_dir):
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
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        ax.plot(t_landmark, ldmkmeasres[:, i], label="residual",
                linestyle="None", marker=".", color=colors[0])
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
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load OD results
    od = load_od_results(results_dir)
    meta        = od["meta"]
    est_states  = od["state_estimates"]
    covariance         = od["covariance"]
    dynamics_residuals = od["dynamics_residuals"]
    landmark_residuals = od["landmark_residuals"]
    bias_mode          = meta.get("bias_mode", "fix_bias")

    print(f"Loaded OD results from {results_dir}")
    print(f"  error_code : {meta.get('error_code')}")
    print(f"  bias_mode  : {bias_mode}")
    print(f"  solver     : {meta.get('solver', {}).get('termination_type')} "
          f"({meta.get('solver', {}).get('num_iterations')} iters, "
          f"cost {meta.get('solver', {}).get('initial_cost', 0):.3f} → "
          f"{meta.get('solver', {}).get('final_cost', 0):.3f})")
    print(f"  states     : {est_states.shape}")

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

    # Covariance is already Nx13 structured (StateResIdx columns)
    est_covars = covariance

    ldmk_t = load_landmark_timestamps(dataset_dir)

    # ── State estimate plots (always; overlaid with GT when available) ───────────
    plot_states(est_states, out_dir,
                true_states=ground_truth if have_gt else None,
                orbit_measurements=measurements if have_gt else None)
    if dynamics_residuals is not None and landmark_residuals is not None:
        plot_residuals_standalone(dynamics_residuals, landmark_residuals,
                                  bias_mode, ldmk_t, out_dir)

    # ── Comparison plots (simulation ground truth only) ───────────────────────
    if have_gt:
        if est_covars is not None:
            plot_errors(ground_truth, est_states, est_covars, bias_mode, out_dir)
        plot_measurements(measurements, ground_truth, est_states, ldmkmeasres, out_dir)
        if ldmkmeasres is not None:
            plot_residuals(lindynres, angdynres, ldmkmeasres,
                           ldmk_t, est_states, ground_truth, bias_mode, out_dir)

    print("Done.")
