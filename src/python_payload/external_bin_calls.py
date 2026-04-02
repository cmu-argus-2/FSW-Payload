"""
This file will implement the code that will call the c++ bins to perform the actions
like record a dataset, or run inference...
"""

import os
import subprocess


def run_prefiltering(img_path, output_folder_path):
    run_path = "/home/argus-payload/Documents/FSW-Payload"
    bin_name = "./bin/RUN_PREFILTERING"

    parsed = {
        "passed": False,
        "is_significant": False,
        "dominant_type": "unknown",
    }

    result = subprocess.run(
        [bin_name, img_path, output_folder_path],
        cwd=run_path,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[prefilter] Binary failed (exit {result.returncode}) for {img_path}")
        print(f"[prefilter] stderr: {result.stderr.strip()}")
        return parsed

    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("Passed:"):
            parsed["passed"] = line.split(":", 1)[1].strip().lower() == "yes"
        elif line.startswith("Significant:"):
            parsed["is_significant"] = line.split(":", 1)[1].strip().lower() == "yes"
        elif line.startswith("Dominant Type:"):
            parsed["dominant_type"] = line.split(":", 1)[1].strip().lower()

    print(f"[prefilter] {Path(img_path).name}: "
          f"passed={parsed['passed']} "
          f"is_significant={parsed['is_significant']} "
          f"dominant_type={parsed['dominant_type']}")

    return parsed

def run_inference(img_path, output_folder_path):
    """
    will run the inference on a given image
    it will also give a path for the generated data to be saved
    for now this will be running on the payload folder that is on Documents
    will move to the run path and run the binary from there 
    """
    
    run_path = "/home/argus-payload/Documents/FSW-Payload"
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
    