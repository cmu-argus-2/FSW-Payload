"""
This file will implement the code that will call the c++ bins to perform the actions
like record a dataset, or run inference...
"""

import os
import subprocess
from pathlib import Path


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
    
    rc_model = "models/trained-rc/V2/rc_model_weights.trt"
    ld_model = "models/trained-ld/V2"
    
    print(f"Running inference on {img_path}")
    print(f"Output folder: {output_folder_path}")
    print(f"RC model: {rc_model}")
    print(f"LD model: {ld_model}")
    
    result = subprocess.run([bin_name, rc_model, ld_model, img_path, output_folder_path],
        cwd=run_path,
        # capture_output=True,
        text=True
    )
    
    # capture result code
    try:
        return_code = result.returncode
        print(f"Return code: {return_code}")
    except Exception as e:
        print(f"Error capturing return code: {e}")
        return_code = -1
    
    return return_code