import argparse
import subprocess
import sys
from pathlib import Path
import os

#!/usr/bin/env python3
"""
Call ../bin/imu_test_gyro with arguments: test_duration and folder location.
"""

import os
import numpy as np
import matplotlib

def main():
    log_folder = os.path.join(os.path.dirname(__file__), "..", "data", "results", "gyro_test")
    duration = 5*60 # 7*60*60  # seconds
    
    gyro_log = Path(log_folder) / "gyro_log.txt"
    try:
        if gyro_log.is_file():
            gyro_log.unlink()
    except Exception as e:
        print(f"Warning: failed to remove existing gyro_log.txt: {e}", file=sys.stderr)
    
    script_dir = os.path.dirname(__file__)
    bin_path = os.path.join(script_dir, "..", "bin", "imu_test_gyro")

    Path(log_folder).mkdir(parents=True, exist_ok=True)

    cmd = [str(bin_path), str(duration), str(log_folder)]
    print("Running:", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as e:
        print("Failed to run binary:", e, file=sys.stderr)
        sys.exit(4)

    # Forward stdout/stderr
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")


    # Call the plotting script
    plots_script = os.path.join(script_dir, "gyro_plot.py")
    if not Path(plots_script).exists():
        print(f"Warning: plotter not found: {plots_script}", file=sys.stderr)
    else:
        cmd_plot = [sys.executable, str(plots_script), str(log_folder)]
        print("Running plotter:", " ".join(cmd_plot))
        try:
            proc_plot = subprocess.run(cmd_plot, check=False, capture_output=True, text=True)
        except Exception as e:
            print("Failed to run gyro_plot.py:", e, file=sys.stderr)
        else:
            if proc_plot.stdout:
                print(proc_plot.stdout, end="")
            if proc_plot.stderr:
                print(proc_plot.stderr, file=sys.stderr, end="")
            if proc_plot.returncode:
                proc = proc_plot
    sys.exit(proc.returncode or 0)


if __name__ == "__main__":
    main()