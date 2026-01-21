from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

#!/usr/bin/env python3
"""
Load HDF5 datasets:
    data/datasets/batch_opt_gen/ground_truth_states.h5
    data/datasets/batch_opt_gen/orbit_measurements.h5
    data/datasets/batch_opt_gen/state_estimates.h5
"""

DATA_DIR = Path("data/datasets/batch_opt_gen")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../data/results/batch_opt")

def load_h5(path: Path) -> dict:
        """Load all datasets from an HDF5 file into a dict {dataset_path: ndarray}."""
        path = Path(path)
        if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
        data = {}
        with h5py.File(path, "r") as f:
                def visitor(name, obj):
                        if isinstance(obj, h5py.Dataset):
                                data[name] = obj[()]  # read into numpy array
                f.visititems(visitor)
        return data

def plot_states(true_states, est_states, orbit_measurements):
    # Plot true and estimated states over time
    true_time = true_states["unixtime"]
    est_time  = est_states["state_estimates"][:,0]
    # Plot position 
    true_pos_eci = true_states["states"][:,:3]
    est_pos_eci = est_states["state_estimates"][:,1:4]
    # Plot position components (3 rows x 1 column): true vs estimated
    t0 = true_time[0] if len(true_time) > 0 else 0
    t_true = true_time - t0
    t_est = est_time - t0

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    comp_labels = ["Position X (km)", "Position Y (km)", "Position Z (km)"]
    colors = ["C0", "C1"]
    for i, ax in enumerate(axs):
        ax.plot(t_true, true_pos_eci[:, i], label="true", color=colors[0])
        ax.plot(t_est, est_pos_eci[:, i], label="est", color=colors[1], linestyle="--")
        ax.set_ylabel(comp_labels[i])
        ax.grid(True)
        if i == 0:
            ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("ECI Position: True vs Estimated")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(RESULTS_DIR, "eci_position_true_vs_estimated.png"))
    
    # Plot velocity
    true_vel_eci = true_states["states"][:,3:6]
    est_vel_eci = est_states["state_estimates"][:,4:7]
    
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    comp_labels = ["Velocity X (km/s)", "Velocity Y (km/s)", "Velocity Z (km/s)"]
    colors = ["C0", "C1"]
    for i, ax in enumerate(axs):
        ax.plot(t_true, true_vel_eci[:, i], label="true", color=colors[0])
        ax.plot(t_est, est_vel_eci[:, i], label="est", color=colors[1], linestyle="--")
        ax.set_ylabel(comp_labels[i])
        ax.grid(True)
        if i == 0:
            ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("ECI Velocity: True vs Estimated")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(RESULTS_DIR, "eci_velocity_true_vs_estimated.png"))
    
    # Plot quaternion
    true_quat = true_states["states"][:,6:10]
    est_quat = est_states["state_estimates"][:,7:11]
    
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 6))
    comp_labels = ["QW", "QX", "QY", "QZ"]
    colors = ["C0", "C1"]
    for i, ax in enumerate(axs):
        ax.plot(t_true, true_quat[:, i], label="true", color=colors[0])
        ax.plot(t_est, est_quat[:, i], label="est", color=colors[1], linestyle="--")
        ax.set_ylabel(comp_labels[i])
        ax.grid(True)
        if i == 0:
            ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("Quaternion: True vs Estimated")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(RESULTS_DIR, "quaternion_true_vs_estimated.png"))
    
    # Plot gyro bias
    true_gyro_bias = true_states["states"][:,13:16]
    est_gyro_bias = est_states["state_estimates"][:,11:14]
    
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    comp_labels = ["Bias X (rad/s)", "Bias Y (rad/s)", "Bias Z (rad/s)"]
    colors = ["C0", "C1"]
    for i, ax in enumerate(axs):
        ax.plot(t_true, true_gyro_bias[:, i], label="true", color=colors[0])
        ax.plot(t_est, est_gyro_bias[:, i], label="est", color=colors[1], linestyle="--")
        ax.set_ylabel(comp_labels[i])
        ax.grid(True)
        if i == 0:
            ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("Gyro Bias: True vs Estimated")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(RESULTS_DIR, "gyro_bias_true_vs_estimated.png"))
    
    # Plot angular velocity
    true_ang_vel = true_states["states"][:,10:13]
    gyro_meas = orbit_measurements['gyro_measurements'][:,1:4]
    t_gyro = orbit_measurements['gyro_measurements'][:,0] - t0
    
    est_ang_vel = gyro_meas - est_gyro_bias
    # est ang vel repeated timestamps every 10 seconds
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    comp_labels = ["Omega X (rad/s)", "Omega Y (rad/s)", "Omega Z (rad/s)"]
    for i, ax in enumerate(axs):
        ax.plot(t_true, true_ang_vel[:, i], label="true", color=colors[0])
        ax.plot(t_est, est_ang_vel[:, i], label="est", color=colors[1], linestyle="--")
        ax.set_ylabel(comp_labels[i])
        ax.grid(True)
        if i == 0:
            ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("Angular Velocity: True vs Estimated")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(RESULTS_DIR, "angular_velocity_true_vs_estimated.png"))
    
    
def plot_errors(true_states, est_states):
    # TODO: Add Covariances
    true_time = true_states["unixtime"]
    est_time  = est_states["state_estimates"][:,0]
    # Plot position 
    true_pos_eci = true_states["states"][:,:3]
    est_pos_eci = est_states["state_estimates"][:,1:4]
    # Plot position components (3 rows x 1 column): true vs estimated
    t0 = true_time[0] if len(true_time) > 0 else 0
    t_true = true_time - t0
    t_est = est_time - t0

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    comp_labels = ["Error X (km)", "Error Y (km)", "Error Z (km)"]
    colors = ["C0", "C1"]
    true_pos_eci_estt = np.zeros(est_pos_eci.shape)
    for i, ax in enumerate(axs):
        true_pos_eci_estt[:,i] = np.interp(t_est, t_true, true_pos_eci[:,i])
        
    for i, ax in enumerate(axs):
        ax.plot(t_est, true_pos_eci_estt[:, i] - est_pos_eci[:, i], 
                                        label="error", color=colors[0])
        ax.set_ylabel(comp_labels[i])
        ax.grid(True)
        if i == 0:
            ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("ECI Position: Error Three Axis")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(RESULTS_DIR, "eci_position_three_error.png"))
    
    pos_norm = np.linalg.norm(true_pos_eci_estt - est_pos_eci, axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_est, pos_norm, label="Position Error Norm", color="C0")
    ax.set_ylabel("Error Norm (km)")
    ax.set_xlabel("time (s) since start")
    ax.set_title("Position Error Norm")
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(RESULTS_DIR, "eci_position_error_norm.png"))
    
    # Plot velocity
    true_vel_eci = true_states["states"][:,3:6]
    est_vel_eci = est_states["state_estimates"][:,4:7]
    
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    comp_labels = ["Error X (km/s)", "Error Y (km/s)", "Error Z (km/s)"]
    colors = ["C0", "C1"]
    true_vel_eci_estt = np.zeros(est_vel_eci.shape)
    for i, ax in enumerate(axs):
        true_vel_eci_estt[:,i] = np.interp(t_est, t_true, true_vel_eci[:,i])

    for i, ax in enumerate(axs):
        ax.plot(t_est, true_vel_eci_estt[:, i] - est_vel_eci[:, i], 
                                        label="error", color=colors[0])
        ax.set_ylabel(comp_labels[i])
        ax.grid(True)
        if i == 0:
            ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("ECI Velocity: Error Three Axis")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(RESULTS_DIR, "eci_velocity_three_error.png"))
    
    vel_norm = np.linalg.norm(true_vel_eci_estt - est_vel_eci, axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_est, vel_norm, label="Velocity Error Norm", color="C0")
    ax.set_ylabel("Error Norm (km/s)")
    ax.set_xlabel("time (s) since start")
    ax.set_title("Velocity Error Norm")
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(RESULTS_DIR, "eci_velocity_error_norm.png"))
    
    # Plot quaternion
    true_quat = true_states["states"][:,6:10]
    est_quat = est_states["state_estimates"][:,7:11]
    
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 6))
    true_quat_estt = np.zeros(est_quat.shape)
    for i, ax in enumerate(axs):
        true_quat_estt[:,i] = np.interp(t_est, t_true, true_quat[:,i])
    comp_labels = ["QW", "QX", "QY", "QZ"]
    colors = ["C0", "C1"]
    for i, ax in enumerate(axs):
        ax.plot(t_est, true_quat_estt[:, i] - est_quat[:, i], label="error", color=colors[0])
        ax.set_ylabel(comp_labels[i])
        ax.grid(True)
        if i == 0:
            ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("Quaternion: Error Four Axis")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(RESULTS_DIR, "quaternion_four_error.png"))
    
    quat_norm = np.linalg.norm(true_quat_estt - est_quat, axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_est, quat_norm, label="Quaternion Error Norm", color="C0")
    ax.set_ylabel("Error Norm")
    ax.set_xlabel("time (s) since start")
    ax.set_title("Quaternion Error Norm")
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(RESULTS_DIR, "quaternion_error_norm.png"))
    
    # Plot gyro bias
    true_gyro_bias = true_states["states"][:,13:16]
    est_gyro_bias = est_states["state_estimates"][:,11:14]
    true_gyro_bias_est = np.zeros(est_gyro_bias.shape)
    
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    for i, ax in enumerate(axs):
        true_gyro_bias_est[:,i] = np.interp(t_est, t_true, true_gyro_bias[:,i])
    comp_labels = ["Bias X (rad/s)", "Bias Y (rad/s)", "Bias Z (rad/s)"]
    colors = ["C0", "C1"]
    for i, ax in enumerate(axs):
        ax.plot(t_est, true_gyro_bias_est[:, i] - est_gyro_bias[:, i], label="error", color=colors[0])
        ax.set_ylabel(comp_labels[i])
        ax.grid(True)
        if i == 0:
            ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("Gyro Bias: Error Three Axis")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(RESULTS_DIR, "gyro_bias_three_error.png"))
    
    gyro_bias_norm = np.linalg.norm(true_gyro_bias_est - est_gyro_bias, axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_est, gyro_bias_norm, label="Gyro Bias Error Norm", color="C0")
    ax.set_ylabel("Error Norm (rad/s)")
    ax.set_xlabel("time (s) since start")
    ax.set_title("Gyro Bias Error Norm")
    ax.grid(True)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(RESULTS_DIR, "gyro_bias_error_norm.png"))


def plot_measurements(measurements, ground_truth_states, state_estimates):
    # plot the measurements vs measurement estimate
    # TODO: Save measurement residuals during batch optimization
    # gyro measurements
    # Plot angular velocity
    true_time = ground_truth_states["unixtime"]
    t0 = true_time[0] if len(true_time) > 0 else 0
    gyro_meas = measurements['gyro_measurements'][:,1:4]
    t_gyro = measurements['gyro_measurements'][:,0] - t0
    colors = ["C0", "C1"]

    # gyromeas = gyromeas_hat
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
    comp_labels = ["Omega X (rad/s)", "Omega Y (rad/s)", "Omega Z (rad/s)"]
    for i, ax in enumerate(axs):
        ax.plot(t_gyro, gyro_meas[:, i], label="est", color=colors[1], linestyle="--")
        ax.set_ylabel(comp_labels[i])
        ax.grid(True)
        if i == 0:
            ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("Gyro measurements")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(RESULTS_DIR, "gyro_measurements.png"))
    
    # landmark measurements
    landmark_meas = measurements['landmark_measurements']
    t_landmark = landmark_meas[:,0] - t0
    
    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(10, 6))
    comp_labels = ["Bear X (km)", "Bear Y (km)", "Bear Z (km)", "Ldmk X (km)", "Ldmk Y (km)", "Ldmk Z (km)"]
    for i, ax in enumerate(axs):
        ax.plot(t_landmark, landmark_meas[:, i+1], color=colors[1], linestyle="None", marker='.') # , markersize=3)
        ax.set_ylabel(comp_labels[i])
        ax.grid(True)
        # if i == 0:
        #     ax.legend(loc="upper right")
    axs[-1].set_xlabel("time (s) since start")
    fig.suptitle("Landmark measurements")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(RESULTS_DIR, "landmark_measurements.png"))

if __name__ == "__main__":
    files = {
            "ground_truth": DATA_DIR / "ground_truth_states.h5",
            "measurements": DATA_DIR / "orbit_measurements.h5",
            "estimates": DATA_DIR / "state_estimates.h5",
    }

    loaded = {}
    for key, path in files.items():
            loaded[key] = load_h5(path)
            print(f"Loaded {key} from {path}: {len(loaded[key])} dataset(s)")
            for ds_name, arr in loaded[key].items():
                    print(f"  {ds_name} -> shape={np.shape(arr)}, dtype={arr.dtype}")

    ground_truth_states = loaded["ground_truth"]
    orbit_measurements = loaded["measurements"]
    state_estimates = loaded["estimates"]
    
    # true and estimated states separately
    plot_states(ground_truth_states, state_estimates, orbit_measurements)
    
    # plot errors
    plot_errors(ground_truth_states, state_estimates)
    
    # plot measurements vs residuals
    plot_measurements(orbit_measurements, ground_truth_states, state_estimates)