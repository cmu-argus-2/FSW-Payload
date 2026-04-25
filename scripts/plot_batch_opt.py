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
import os
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

    res_path = results_dir / "residuals.csv"
    residuals = np.genfromtxt(res_path, delimiter=",", skip_header=1) if res_path.exists() else None

    return {
        "meta": meta,
        "state_estimates": state_estimates,   # Nx14 array
        "covariance": covariance,             # 1-D array or None
        "residuals": residuals,               # 1-D array or None
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


# ── Processing helpers ─────────────────────────────────────────────────────────

def process_covariances(covariances, bias_mode):
    state_count_with_bias    = 12
    state_count_without_bias = 9
    if bias_mode == "no_bias":
        nsteps = covariances.shape[0] // state_count_without_bias
        est_covariances = covariances.reshape(-1, nsteps, state_count_without_bias)
    elif bias_mode == "fix_bias":
        start    = covariances[:state_count_with_bias]
        rest     = covariances[state_count_with_bias:]
        rest_cov = rest.reshape(-1, state_count_without_bias)
        rest_cov = np.hstack((
            rest_cov,
            start[state_count_without_bias:].reshape(1, 3).repeat(rest_cov.shape[0], axis=0)
        ))
        est_covariances = np.vstack((start, rest_cov))
    elif bias_mode == "tv_bias":
        nsteps = covariances.shape[0] // state_count_with_bias
        est_covariances = covariances.reshape(-1, nsteps, state_count_with_bias)
    else:
        raise ValueError(f"Unknown bias mode: {bias_mode}")
    return est_covariances


def process_residuals(residuals, n_steps, n_ldmks, bias_mode):
    lindynres = np.zeros((n_steps - 1, 6))
    for i in range(n_steps - 1):
        lindynres[i] = residuals[i * 6:(i + 1) * 6]
    k = (n_steps - 1) * 6

    if bias_mode in ["fix_bias", "no_bias"]:
        angdynres = np.zeros((n_steps - 1, 3))
        for i in range(n_steps - 1):
            angdynres[i] = residuals[k + i * 3:k + (i + 1) * 3]
        k += (n_steps - 1) * 3
    elif bias_mode == "tv_bias":
        angdynres = np.zeros((n_steps - 1, 6))
        for i in range(n_steps - 1):
            angdynres[i] = residuals[k + i * 6:k + (i + 1) * 6]
        k += (n_steps - 1) * 6
    else:
        raise ValueError(f"Unknown bias mode: {bias_mode}")

    ldmkmeasres = np.zeros((n_ldmks, 3))
    for i in range(n_ldmks):
        ldmkmeasres[i] = residuals[k + i * 3:k + (i + 1) * 3]

    return lindynres, angdynres, ldmkmeasres


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_states(true_states, est_states, orbit_measurements, out_dir):
    true_time   = true_states["unixtime"]
    est_time    = est_states[:, 0]
    t0          = true_time[0] if len(true_time) > 0 else 0
    t_true      = true_time - t0
    t_est       = est_time - t0
    colors      = ["C0", "C1"]

    # Position
    true_pos = true_states["states"][:, :3]
    est_pos  = est_states[:, 1:4]
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        ax.plot(t_true, true_pos[:, i], label="true", color=colors[0])
        ax.plot(t_est,  est_pos[:, i],  label="est",  color=colors[1], linestyle="--")
        ax.set_ylabel(["Position X (km)", "Position Y (km)", "Position Z (km)"][i])
        ax.grid(True)
        if i == 0: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("ECI Position: True vs Estimated")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "eci_position_true_vs_estimated.png")
    plt.close()

    # Velocity
    true_vel = true_states["states"][:, 3:6]
    est_vel  = est_states[:, 4:7]
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        ax.plot(t_true, true_vel[:, i], label="true", color=colors[0])
        ax.plot(t_est,  est_vel[:, i],  label="est",  color=colors[1], linestyle="--")
        ax.set_ylabel(["Velocity X (km/s)", "Velocity Y (km/s)", "Velocity Z (km/s)"][i])
        ax.grid(True)
        if i == 0: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("ECI Velocity: True vs Estimated")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "eci_velocity_true_vs_estimated.png")
    plt.close()

    # Quaternion — est order in CSV: quat_x, quat_y, quat_z, quat_w (cols 7-10)
    # reorder to [w, x, y, z] for display
    true_quat = true_states["states"][:, 6:10]
    true_quat = true_quat * np.sign(true_quat[:, 0:1])
    est_quat  = est_states[:, [10, 7, 8, 9]]   # [w, x, y, z]
    est_quat  = est_quat * np.sign(est_quat[:, 0:1])
    fig, axs  = plt.subplots(4, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        ax.plot(t_true, true_quat[:, i], label="true", color=colors[0])
        ax.plot(t_est,  est_quat[:, i],  label="est",  color=colors[1], linestyle="--")
        ax.set_ylabel(["QW", "QX", "QY", "QZ"][i])
        ax.grid(True)
        if i == 0: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("Quaternion: True vs Estimated")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "quaternion_true_vs_estimated.png")
    plt.close()

    # Gyro bias
    true_bias = true_states["states"][:, 13:16]
    est_bias  = est_states[:, 11:14]
    fig, axs  = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        ax.plot(t_true, true_bias[:, i], label="true", color=colors[0])
        ax.plot(t_est,  est_bias[:, i],  label="est",  color=colors[1], linestyle="--")
        ax.set_ylabel(["Bias X (rad/s)", "Bias Y (rad/s)", "Bias Z (rad/s)"][i])
        ax.grid(True)
        if i == 0: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("Gyro Bias: True vs Estimated")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "gyro_bias_true_vs_estimated.png")
    plt.close()

    # Angular velocity (gyro measurement minus estimated bias)
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
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "angular_velocity_true_vs_estimated.png")
    plt.close()


def plot_errors(true_states, est_states, est_covars, bias_mode, out_dir):
    true_time = true_states["unixtime"]
    est_time  = est_states[:, 0]
    t0        = true_time[0] if len(true_time) > 0 else 0
    t_true    = true_time - t0
    t_est     = est_time - t0
    colors    = ["C0", "C1"]

    # Position error
    true_pos           = true_states["states"][:, :3]
    est_pos            = est_states[:, 1:4]
    true_pos_at_est    = np.zeros(est_pos.shape)
    est_covars_pos     = np.sqrt(est_covars[:, :3])
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
    est_covars_pos_norm = np.sqrt(est_covars[:, :3].sum(axis=1))
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
    vel_covar       = np.sqrt(est_covars[:, 3:6])
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
    est_covars_vel_norm = np.sqrt(est_covars[:, 3:6].sum(axis=1))
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

    att_covar        = np.rad2deg(np.sqrt(est_covars[:, 6:9]))
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
                       else np.sqrt(est_covars[:, 9:12]))
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
    est_covars_gyro_bias_norm = np.sqrt(est_covars[:, 9:12].sum(axis=1))
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


def plot_residuals(lindynres, angdynres, ldmkmeasres, measurements, est_states,
                   true_states, bias_mode, out_dir):
    true_time = true_states["unixtime"]
    est_time  = est_states[:, 0]
    t0        = true_time[0] if len(true_time) > 0 else 0
    t_est     = est_time - t0
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
    plt.savefig(out_dir / "linear_dynamics_residuals.png")
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
    axs[-1].set_xlabel("time step")
    fig.suptitle("Angular Dynamics Residuals")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "angular_dynamics_residuals.png")
    plt.close()

    landmark_meas = measurements["landmark_measurements"]
    t_landmark    = landmark_meas[:, 0] - t0
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        ax.plot(t_landmark, ldmkmeasres[:, i], label="residual",
                linestyle="None", marker=".", color=colors[0])
        ax.fill_between(t_landmark, -3, 3, color="C0", alpha=0.3, label="3-sigma")
        ax.set_ylabel(["Ldmk Meas Res X", "Ldmk Meas Res Y", "Ldmk Meas Res Z"][i]); ax.grid(True)
        if i == 0: ax.legend(loc="upper right")
    axs[-1].set_xlabel("time step")
    fig.suptitle("Landmark Measurement Residuals")
    plt.tight_layout(); plt.subplots_adjust(top=0.92)
    plt.savefig(out_dir / "landmark_measurement_residuals.png")
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
    covariance  = od["covariance"]
    residuals   = od["residuals"]
    bias_mode   = meta.get("bias_mode", "fix_bias")

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
    if residuals is not None and have_gt:
        n_steps  = est_states.shape[0]
        n_ldmks  = measurements["landmark_measurements"].shape[0]
        lindynres, angdynres, ldmkmeasres = process_residuals(
            residuals, n_steps, n_ldmks, bias_mode)
    else:
        lindynres = angdynres = ldmkmeasres = None

    # Covariance processing
    if covariance is not None:
        est_covars = process_covariances(covariance, bias_mode)
        est_covars = np.squeeze(est_covars)
    else:
        est_covars = None
        print("Covariance unavailable — skipping error-bound plots.")

    if have_gt:
        plot_states(ground_truth, est_states, measurements, out_dir)
        if est_covars is not None:
            plot_errors(ground_truth, est_states, est_covars, bias_mode, out_dir)
        plot_measurements(measurements, ground_truth, est_states, ldmkmeasres, out_dir)
        if ldmkmeasres is not None:
            plot_residuals(lindynres, angdynres, ldmkmeasres,
                           measurements, est_states, ground_truth, bias_mode, out_dir)

    print("Done.")
