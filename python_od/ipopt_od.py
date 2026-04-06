#!/usr/bin/env python3
"""
Batch orbit determination using CasADi + IPOPT — fixed gyro bias.

Reproduces the fixed-bias variant of the C++ Ceres batch optimizer
(scripts/navigation/run_batch_opt.cpp).

Run from the repository root:
    python python_od/ipopt_od.py

Or from within python_od/:
    python ipopt_od.py
"""
import sys
from pathlib import Path

import numpy as np

# Allow running from either the repo root or the python_od/ directory.
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_h5
from optimizer import build_and_solve, save_results

# ── Paths ───────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/datasets/batch_opt_gen")
OUTPUT_DIR = DATA_DIR


def main() -> None:
    # ── Load measurements ───────────────────────────────────────────────────────
    meas = load_h5(DATA_DIR / "orbit_measurements.h5")

    landmark_measurements = meas["landmark_measurements"].astype(np.float64)    # (N_lmk,  7)
    landmark_group_starts = meas["group_starts"].astype(bool).ravel()  # (N_lmk,)
    gyro_measurements     = meas["gyro_measurements"].astype(np.float64)         # (N_gyro, 4)

    print(f"Landmark measurements : {landmark_measurements.shape}")
    print(f"Gyro measurements     : {gyro_measurements.shape}")
    print(f"Landmark groups       : {landmark_group_starts.sum()}")

    # ── Solve ───────────────────────────────────────────────────────────────────
    results = build_and_solve(
        landmark_measurements,
        landmark_group_starts,
        gyro_measurements,
        max_dt=60.0,
    )

    # ── Report ──────────────────────────────────────────────────────────────────
    print(f"\nEstimated gyro bias : {results['gyro_bias']} rad/s")
    print(f"Final position      : {results['positions'][-1]} km")
    print(f"Final velocity      : {results['velocities'][-1]} km/s")

    # ── Save ────────────────────────────────────────────────────────────────────
    save_results(results, OUTPUT_DIR / "state_estimates.h5")


if __name__ == "__main__":
    main()
