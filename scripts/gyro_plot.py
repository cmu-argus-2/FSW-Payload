import os
from typing import Tuple, List
import numpy as np
import allantools as at
import matplotlib.pyplot as plt


def load_gyro_log() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load gyro_log.txt from the same folder as this script.
    Expected line format:
      2025-11-25 00:00:11 -0.27439 -0.171494 0.537348
    Returns:
      times (np.ndarray of datetime64[s]), gyro_x, gyro_y, gyro_z (np.ndarray[float])
    """
    base = os.path.dirname(__file__)
    base = os.path.join(base, "..","data", "results", "gyro_test")
    path = os.path.join(base, "gyro_log.txt")
        
    time_strs: List[str] = []
    gx_list: List[float] = []
    gy_list: List[float] = []
    gz_list: List[float] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                # ignore malformed lines
                continue
            date, tm = parts[0], parts[1]
            try:
                gx, gy, gz = map(float, parts[2:5])
            except ValueError:
                continue
            time_strs.append(f"{date} {tm}")
            gx_list.append(gx)
            gy_list.append(gy)
            gz_list.append(gz)

    times = np.array(time_strs, dtype="datetime64[s]")
    gyro_x = np.array(gx_list, dtype=float)
    gyro_y = np.array(gy_list, dtype=float)
    gyro_z = np.array(gz_list, dtype=float)
    return times, gyro_x, gyro_y, gyro_z


def plot_gyro(time, gyro_x, gyro_y, gyro_z):
    if time.size == 0:
        return
    elapsed = (time - time[0]).astype('timedelta64[s]').astype(int)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    axes[0].plot(elapsed, gyro_x, color="C0")
    axes[0].set_ylabel("gyro_x")
    axes[0].grid(True)

    axes[1].plot(elapsed, gyro_y, color="C1")
    axes[1].set_ylabel("gyro_y")
    axes[1].grid(True)

    axes[2].plot(elapsed, gyro_z, color="C2")
    axes[2].set_ylabel("gyro_z")
    axes[2].set_xlabel("seconds elapsed")
    axes[2].grid(True)

    fig.suptitle("Gyroscope readings")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    result_folder = os.path.join(os.path.dirname(__file__), "..", "data", "results")
    out_dir = os.path.join(result_folder, "gyro_test")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "gyro_plot.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    

def print_gyro_stats(gyro_x, gyro_y, gyro_z):
    if gyro_x.size == 0:
        print("No gyro data")
        return
    for label, arr in (("gyro_x", gyro_x), ("gyro_y", gyro_y), ("gyro_z", gyro_z)):
        mean = np.mean(arr)
        std = np.std(arr)
        print(f"{label}: mean={mean:.6f}, std={std:.6f}")


def compute_and_plot_allan(time, gyro_x, gyro_y, gyro_z):
    if time.size < 2 or gyro_x.size == 0:
        print("Not enough data for Allan deviation")
        return

    dt = (time[1] - time[0]).astype('timedelta64[s]').astype(float)
    (taus_x, adev_x, adeverr_x, n_x) = at.oadev(gyro_x, rate=1.0 / dt, data_type='phase', taus="all")
    (taus_y, adev_y, adeverr_y, n_y) = at.oadev(gyro_y, rate=1.0 / dt, data_type='phase', taus="all")
    (taus_z, adev_z, adeverr_z, n_z) = at.oadev(gyro_z, rate=1.0 / dt, data_type='phase', taus="all")

    # plot log-log
    if taus_x.size == 0 and taus_y.size == 0 and taus_z.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if taus_x.size:
        ax.loglog(taus_x, adev_x, "-o", label="gyro_x", color="C0", markersize=4)
        ax.fill_between(taus_x, adev_x+adeverr_x, adev_x-adeverr_x, 
                        alpha=0.2, color='C0', label='Filled Area')

    if taus_y.size:
        ax.loglog(taus_y, adev_y, "-o", label="gyro_y", color="C1", markersize=4)
        ax.fill_between(taus_y, adev_y+adeverr_y, adev_y-adeverr_y, 
                        alpha=0.2, color='C1', label='Filled Area')
    if taus_z.size:
        ax.loglog(taus_z, adev_z, "-o", label="gyro_z", color="C2", markersize=4)
        ax.fill_between(taus_z, adev_z+adeverr_z, adev_z-adeverr_z, 
                        alpha=0.2, color='C2', label='Filled Area')

    ax.set_xlabel("tau (s)")
    ax.set_ylabel("Allan deviation")
    ax.grid(which="both", linestyle=":", linewidth=0.5)
    ax.legend()
    fig.tight_layout()

    result_folder = os.path.join(os.path.dirname(__file__), "..", "data", "results")
    out_dir = os.path.join(result_folder, "gyro_test")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "gyro_allan_deviation.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Allan deviation plot to {out_png}")


if __name__ == "__main__":
    time, gyro_x, gyro_y, gyro_z = load_gyro_log()
    print(f"Loaded {len(time)} samples")
    plot_gyro(time, gyro_x, gyro_y, gyro_z)
    print_gyro_stats(gyro_x, gyro_y, gyro_z)
    compute_and_plot_allan(time, gyro_x, gyro_y, gyro_z)