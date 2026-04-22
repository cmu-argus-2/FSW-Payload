"""
This file will implement the code that will call the c++ bins to perform the actions
like record a dataset, or run inference...
"""

import os
import subprocess
import time
from pathlib import Path

_dataset_process: subprocess.Popen | None = None
_dataset_folder: Path | None = None


def run_prefiltering(img_path, output_folder_path):
    run_path = "/home/argus/Documents/FSW-Payload"
    bin_name = "./bin/RUN_PREFILTER"
    
    # resolve to absolute so the binary can find it regardless of cwd
    img_path = str(Path(img_path).resolve())
    output_folder_path = str(Path(output_folder_path).resolve())
    
    parsed = {
        "passed": False,
        "is_significant": False,
        "dominant_type": "unknown",
        "avg_hue": 0.0,
        "avg_saturation": 0.0,
        "avg_value": 0.0,
        "color_std": 0.0,
        "contrast_std": 0.0,
        "cloudiness": 0,
        "avg_rgb": [0.0, 0.0, 0.0],
    }

    result = subprocess.run(
        [bin_name, img_path, output_folder_path],
        cwd=run_path,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[prefilter] Binary exited with code {result.returncode} (continuing to parse stdout)")
        print(f"[prefilter] stderr: {result.stderr.strip()}")

    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("Passed:"):
            parsed["passed"] = line.split(":", 1)[1].strip().lower() == "yes"
        elif line.startswith("Significant:"):
            parsed["is_significant"] = line.split(":", 1)[1].strip().lower() == "yes"
        elif line.startswith("Dominant Type:"):
            parsed["dominant_type"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("Hue:"):
            parsed["avg_hue"] = float(line.split(":", 1)[1].split("(")[0].strip())
        elif line.startswith("Saturation:"):
            parsed["avg_saturation"] = float(line.split(":", 1)[1].split("(")[0].strip())
        elif line.startswith("Brightness:"):
            parsed["avg_value"] = float(line.split(":", 1)[1].split("(")[0].strip())
        elif line.startswith("Color Std:"):
            parsed["color_std"] = float(line.split(":", 1)[1].split("(")[0].strip())
        elif line.startswith("Contrast Std:"):
            parsed["contrast_std"] = float(line.split(":", 1)[1].split("(")[0].strip())
        elif line.startswith("Cloudiness:"):
            parsed["cloudiness"] = int(line.split(":", 1)[1].split("%")[0].strip())
        elif line.startswith("Avg RGB:"):
            vals = line.split("(")[1].split(")")[0].split(",")
            parsed["avg_rgb"] = [float(v.strip()) for v in vals]

    print(
        f"[prefilter] {Path(img_path).name}: "
        f"passed={parsed['passed']} is_significant={parsed['is_significant']} dominant_type={parsed['dominant_type']} | "
        f"hue={parsed['avg_hue']:.1f} sat={parsed['avg_saturation']:.1f} brightness={parsed['avg_value']:.1f} "
        f"color_std={parsed['color_std']:.1f} contrast_std={parsed['contrast_std']:.1f} "
        f"cloudiness={parsed['cloudiness']}% avg_rgb=({parsed['avg_rgb'][0]:.0f},{parsed['avg_rgb'][1]:.0f},{parsed['avg_rgb'][2]:.0f})"
    )

    return parsed

def run_inference(img_path, output_folder_path):
    """
    will run the inference on a given image
    it will also give a path for the generated data to be saved
    for now this will be running on the payload folder that is on Documents
    will move to the run path and run the binary from there 
    """
    
    # run_path = "/home/argus-payload/Documents/FSW-Payload"
    run_path = "."
    bin_name = "./bin/RUN_INFERENCE"
    
    rc_model = "models/V1/trained-rc/effnet_0997acc.trt"
    ld_model = "models/V1/trained-ld"
    
    print(f"Running inference on {img_path}")
    print(f"Output folder: {output_folder_path}")
    print(f"RC model: {rc_model}")
    print(f"LD model: {ld_model}")
    
    result = subprocess.run([bin_name, rc_model, ld_model, img_path, output_folder_path],
        cwd=run_path,
        # capture_output=True,
        text=True
    )
    
    # TODO check if it has completed successfully
    
    return True

def start_dataset(args: list[str] | None = None) -> tuple[bool, Path | None]:
    """
    Start the dataset collection binary as a background subprocess.
    
    args: optional list of CLI arguments to pass (e.g. ["-n", "10", "-m", "30.0"])
    
    Returns: tuple of (success: bool, dataset_folder: Path | None)
             - success: True if started successfully, False if already running
             - dataset_folder: Path object for datasets/<capture_start_ms>/
    """
    global _dataset_process, _dataset_folder
    if _dataset_process is not None and _dataset_process.poll() is None:
        return False, None  # already running

    # Capture the timestamp in milliseconds to match C++ naming convention
    # Pass it to C++ via -t flag so both Python and C++ use the exact same timestamp
    capture_start_ms = int(time.time() * 1000)
    _dataset_folder = Path("datasets") / str(capture_start_ms)

    run_path = "."
    bin_name = "./bin/test_dataset_collection"
    cmd = [bin_name, "-t", str(capture_start_ms)] + (args or [])
    _dataset_process = subprocess.Popen(cmd, cwd=run_path, text=True)
    return True, _dataset_folder


def get_dataset_folder() -> Path | None:
    """
    Get the current dataset folder path.
    Returns the Path object for the active dataset, or None if no dataset is running.
    """
    global _dataset_folder
    return _dataset_folder


def stop_dataset() -> bool:
    """
    Stop the dataset collection subprocess if it is running.
    Returns True if it was stopped, False if it was not running.
    """
    global _dataset_process, _dataset_folder
    if _dataset_process is None or _dataset_process.poll() is not None:
        _dataset_process = None
        _dataset_folder = None
        return False  # not running

    _dataset_process.terminate()
    try:
        _dataset_process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        _dataset_process.kill()
        _dataset_process.wait()

    _dataset_process = None
    _dataset_folder = None
    return True


def is_dataset_running() -> bool:
    """Returns True if the dataset subprocess is currently running."""
    return _dataset_process is not None and _dataset_process.poll() is None