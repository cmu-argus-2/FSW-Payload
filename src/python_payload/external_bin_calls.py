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


def run_dataset_collection(camera_bit_flag, capture_rate, imu_hz, duration):
    """
    This will call the binary to perform the dataset collection
    it will generate a dataset.json file that will be send to the mainboard
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
        "active_cameras": [bool(camera_bit_flag & (1 << i)) for i in range(4)]
    }
    
    with open(os.path.join("config", "dataset_config.toml"), 'w') as f:
        toml.dump(toml_dict, f)
    
    
    # TODO: do I need to cast to string?
    try:
        result = subprocess.run([bin_name],
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
    
    dataset_path = path_out_file.read_text().strip() + "dataset.json"
    print(f"Test dataset generated at: {dataset_path}")
    return dataset_path

def run_dataset_processing(dataset_path, level_processing, rc_version, ld_version):
    """
    This will call the binary to perform the dataset processing
    it will generate a processing.json that will be send to the mainboard

    here we do not need to write the toml file because they are read arguments
    
    TODO: it should have a timeout as well 
    """

    run_path = "."
    bin_name = "./bin/reprocess_dataset"
    
    print(f"Running dataset processing on {dataset_path}")
    print(f"Level of processing: {level_processing}")
    print(f"RC version: {rc_version}")
    print(f"LD version: {ld_version}")
    
    result = subprocess.run([bin_name, dataset_path, str(level_processing), "1", str(rc_version), str(ld_version)],
        cwd=run_path,
        # capture_output=True,
        text=True
    )
    
    try:
        return_code = result.returncode
        print(f"Return code: {return_code}")
    except Exception as e:
        print(f"Error capturing return code: {e}")
        return None
    
    # TODO: it is writing to path.out, but  I am actually not going to use it
    # i will be assuming the dataset_path that was sent as argument
    
    json_path = os.path.join(dataset_path, "processing.json")
    
    # for now we will just return the same path
    return json_path


def run_orbit_determination(dataset_path, max_iter, max_runtime):
    """
    This will call the binary to perform the orbit determination
    it will generate a results.json that will be send to the mainboard

    for now this will just be a placeholder that will return the same path

    here we do not need to write the toml file because they are read arguments
    """
    
    timeout = max_runtime + 10
    run_path = "."
    bin_name = "./bin/run_dataset"
    
    
    try:
        result = subprocess.run([bin_name, "--dataset_path", dataset_path, "--max_iterations", str(max_iter), "--max_runtime", str(max_runtime)],
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
    
    # TODO: it is writing to path.out, but  I am actually not going to use it
    # i will be assuming the dataset_path that was sent as argument
    
    json_path = os.path.join(dataset_path, "results.json")
    
    # for now we will just return the same path
    return json_path