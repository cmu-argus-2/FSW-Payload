#!/usr/bin/env python3
"""
Batch orbit determination using CasADi + IPOPT.

  • Reads landmark_measurements.csv and imu_data.csv from a dataset folder
  • Runs the fixed-bias batch OD optimizer
  • Writes CSV results to data/results/<unix_ms>/

Usage (from the repository root):
    python python_od/ipopt_od.py <dataset_folder> [--j2] [--drag]

Example:
    python python_od/ipopt_od.py data/datasets/17R_Florida_nadir_test
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from data_loader import load_landmark_csv, load_imu_csv
from optimizer import build_and_solve_with_outlier_rejection, save_results
from residuals import IntegratorType


def main() -> None:
    parser = argparse.ArgumentParser(description="Python batch OD (CasADi/IPOPT)")
    parser.add_argument("dataset_folder", type=Path,
                        help="Path to dataset folder containing "
                             "landmark_measurements.csv and imu_data.csv")
    parser.add_argument("--j2",   action="store_true", default=True, help="Enable J2 perturbation")
    parser.add_argument("--drag", action="store_true", default=True, help="Enable atmospheric drag")
    parser.add_argument("--rk4",  action="store_true", default=True, help="Use RK4 integrator (default: Euler)")
    args = parser.parse_args()

    dataset_folder = Path(args.dataset_folder)
    run_unix_ms    = int(time.time() * 1000)

    # ── Load measurements ───────────────────────────────────────────────────────
    lm_csv  = dataset_folder / "landmark_measurements.csv"
    imu_csv = dataset_folder / "imu_data.csv"

    print(f"Loading landmark measurements from {lm_csv} …")
    landmark_measurements, landmark_group_starts, landmark_uncertainties = \
        load_landmark_csv(lm_csv)

    print(f"Loading IMU data from {imu_csv} …")
    gyro_measurements = load_imu_csv(imu_csv)

    print(f"Landmark rows   : {len(landmark_measurements)}")
    print(f"Landmark groups : {landmark_group_starts.sum()}")
    print(f"Gyro rows       : {len(gyro_measurements)}")

    # ── Results folder: data/results/<unix_ms> ─────────────────────────────────
    results_dir = Path("data/results") / str(run_unix_ms)

    # ── Solve ───────────────────────────────────────────────────────────────────
    integrator = IntegratorType.RK4 if args.rk4 else IntegratorType.FORWARD_EULER

    t0 = time.time()
    results = build_and_solve_with_outlier_rejection(
        landmark_measurements,
        landmark_group_starts,
        gyro_measurements,
        landmark_uncertainties = landmark_uncertainties,
        # uma_std         = 1e-5,
        integrator_type    = integrator,
        use_j2             = args.j2,
        use_drag           = args.drag,
        cd_nominal         = 2.2,
        cd_std             = 1.0 if args.drag else None,
        compute_covariance = True,
        mahal_threshold    = 5.0,
        max_iterations     = 10,
        landmark_huber_M   = 3.0,
    )
    run_time_ms = int((time.time() - t0) * 1000)

    # ── Report ──────────────────────────────────────────────────────────────────
    bias = np.asarray(results["gyro_bias"]).ravel()
    print(f"\nEstimated gyro bias : {bias} rad/s")
    print(f"Final position      : {results['positions'][-1]} km")
    print(f"Final velocity      : {results['velocities'][-1]} km/s")

    # ── Save ────────────────────────────────────────────────────────────────────
    meta = {
        "dataset_folder":        str(dataset_folder),
        "run_unix_ms":           run_unix_ms,
        "run_time_ms":           run_time_ms,
        "use_j2":                args.j2,
        "use_drag":              args.drag,
        "integrator":            integrator.value,
        "inputs": {
            "num_landmark_rows":   int(len(landmark_measurements)),
            "num_landmark_groups": int(landmark_group_starts.sum()),
            "num_gyro_rows":       int(len(gyro_measurements)),
        },
        "outputs": {
            "num_state_estimates": int(len(results["state_timestamps"])),
            "covariance_available": "pos_var" in results,
        },
    }
    save_results(results, results_dir, meta=meta)


if __name__ == "__main__":
    main()
