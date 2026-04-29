"""
This file will implement the code that will call the c++ bins to perform the actions
like record a dataset, or run inference...
"""

import os
import subprocess
import sys
from pathlib import Path
import toml




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


def run_dataset_collection(imu_hz, capture_rate, duration):
    """
    This will call the binary to perform the dataset collection
    The binary should return the path to the dataset.json file
    this file will be sent to the ground

    it will have a timeout of duration + 20 seconds to make sure it does not run indefinitely

    to pass the parameters I will be writting to the dataset_config.toml    
    """
    
    timeout = duration + 10
    run_path = "."
    bin_name = "./bin/run_dataset"
    
    
    # write the config file
    toml_dict = {
        "imu_sample_rate_hz": imu_hz,
        "image_capture_rate": capture_rate,
        "maximum_period": duration,
        "active_cameras": ["true", "false", "false", "false"]  
    }
    
    with open(os.path.join("config", "dataset_config.toml"), 'w') as f:
        toml.dump(toml_dict, f)
    
    
    # TODO: do I need to cast to string?
    try:
        result = subprocess.run([bin_name, str(imu_hz), str(capture_rate), str(duration)],
            cwd=run_path,
            # capture_output=True,
            timeout=timeout,
            text=True
        )
    except subprocess.TimeoutExpired:
        print(f"Dataset collection timed out after {timeout} seconds")
        return None
    
    try:
        return_code = result.returncode
        print(f"Return code: {return_code}")
    except Exception as e:
        print(f"Error capturing return code: {e}")
        return None
    
    # lets read the json path from the output file
    path_out_file = Path("path.out")
    if not path_out_file.exists():
        print("Error: path.out file not created")
        return None
    
    dataset_path = path_out_file.read_text().strip()
    print(f"Test dataset generated at: {dataset_path}")
    return dataset_path
