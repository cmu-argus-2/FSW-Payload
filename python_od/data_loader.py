"""
Data loading and state timestamp generation for the batch orbit determination problem.
Mirrors the C++ get_state_timestamps() logic in src/navigation/batch_optimization.cpp.
"""
from pathlib import Path
import math

import h5py
import numpy as np


def load_h5(path: Path) -> dict:
    """Load all datasets from an HDF5 file into a dict {dataset_name: ndarray}."""
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


def get_state_timestamps(
    landmark_measurements: np.ndarray,   # (N_lmk, 7): [t, bx, by, bz, lx, ly, lz]
    landmark_group_starts: np.ndarray,   # (N_lmk,) bool — True at the start of each group
    gyro_measurements: np.ndarray,       # (N_gyro, 4): [t, wx, wy, wz]
    max_dt: float,
) -> tuple[list[float], list[int], list[int]]:
    """
    Merge landmark group and gyro timestamps into a unified state timeline.
    Intermediate states are inserted wherever the gap between consecutive
    timestamps exceeds max_dt, equally spaced so every sub-interval <= max_dt.

    Returns:
        state_timestamps:       list[float]  — unified state timeline
        landmark_group_indices: list[int]    — index into state_timestamps for each landmark group
        gyro_indices:           list[int]    — index into state_timestamps for each gyro measurement
    """
    state_timestamps: list[float] = []
    landmark_group_indices: list[int] = []
    gyro_indices: list[int] = []

    n_lmk  = len(landmark_measurements)
    n_gyro = len(gyro_measurements)

    # Pointers into the measurement arrays
    next_lmk  = 0   # always points at a group-start row (or past end)
    next_gyro = 0

    def _append(t: float) -> None:
        """Append t, inserting evenly-spaced intermediates if dt > max_dt."""
        if not state_timestamps:
            state_timestamps.append(t)
            return
        last = state_timestamps[-1]
        dt = t - last
        if dt <= max_dt:
            if dt > 1e-9:
                state_timestamps.append(t)
            # dt <= 1e-9: near-duplicate timestamp — skip
        else:
            n_steps = math.ceil(dt / max_dt)
            for step in range(1, n_steps + 1):
                state_timestamps.append(last + (step / n_steps) * dt)

    def _consume_landmark() -> None:
        nonlocal next_lmk
        _append(float(landmark_measurements[next_lmk, 0]))
        landmark_group_indices.append(len(state_timestamps) - 1)
        # Advance to next group-start row
        next_lmk += 1
        while next_lmk < n_lmk and not landmark_group_starts[next_lmk]:
            next_lmk += 1

    def _consume_gyro() -> None:
        nonlocal next_gyro
        _append(float(gyro_measurements[next_gyro, 0]))
        gyro_indices.append(len(state_timestamps) - 1)
        next_gyro += 1

    while True:
        lmk_valid  = next_lmk  < n_lmk
        gyro_valid = next_gyro < n_gyro

        if not lmk_valid and not gyro_valid:
            break

        if lmk_valid and gyro_valid:
            t_lmk  = float(landmark_measurements[next_lmk,  0])
            t_gyro = float(gyro_measurements[next_gyro, 0])
            if t_lmk <= t_gyro:
                _consume_landmark()
            else:
                _consume_gyro()
        elif lmk_valid:
            _consume_landmark()
        else:
            _consume_gyro()

    return state_timestamps, landmark_group_indices, gyro_indices
