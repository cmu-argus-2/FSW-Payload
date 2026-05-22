"""
This file will implement the code that will call the c++ bins to perform the actions
like record a dataset, or run inference...
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path
import toml


def _dataset_folder_path(dataset_path):
    path = Path(dataset_path)
    if path.name in ("dataset.json", "processing.json"):
        return str(path.parent)
    return str(path)


def run_inference(img_path, output_folder_path, level_processing=3, rc_version=5, ld_version=3):
    """
    will run the inference on a given image
    it will also give a path for the generated data to be saved
    for now this will be running on the payload folder that is on Documents
    will move to the run path and run the binary from there 
    """
    
    # run_path = "/home/argus-payload/Documents/FSW-Payload"
    run_path = "."
    bin_name = "./bin/reprocess_image"
    output_folder = Path(output_folder_path) if output_folder_path else Path(".")
    output_folder.mkdir(parents=True, exist_ok=True)
    out_path = output_folder / "reprocess_image_path.out"
    
    print(f"Running inference on {img_path}")
    print(f"Output folder: {output_folder_path}")
    print(f"Level of processing: {level_processing}")
    print(f"RC version: {rc_version}")
    print(f"LD version: {ld_version}")
    
    result = subprocess.run([bin_name, img_path,
                             "--target-stage", str(level_processing),
                             "--overwrite",
                             "--rc-version", str(rc_version),
                             "--ld-version", str(ld_version),
                             "--bypass-prefilter-rejection",
                             "--out", str(out_path)],
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
    
    if return_code != 0:
        return None

    path_out_file = Path(out_path)
    if not path_out_file.exists():
        print(f"Error: {path_out_file} file not created")
        return None

    frame_json_path = path_out_file.read_text().strip()
    if not frame_json_path:
        print(f"Error: reprocess_image output file is empty: {path_out_file}")
        return None

    print(f"Frame metadata generated at: {frame_json_path}")
    return frame_json_path


def run_dataset_collection(camera_bit_flag, capture_rate, imu_hz, duration, camera_params):
    """
    This will call the binary to perform the dataset collection
    it will generate a dataset.json file that will be send to the mainboard
    this file will be sent to the ground

    it will have a timeout of duration + 20 seconds to make sure it does not run indefinitely

    to pass the parameters I will be writting to the dataset_config.toml    
    """
    
    timeout = duration + 20
    run_path = "."
    bin_name = "./bin/run_dataset"
    
    
    config_file_path = os.path.join("config", "dataset_config.toml")
    toml_dict = toml.load(config_file_path)
    # write the config file
    toml_dict["imu_sample_rate_hz"] = imu_hz
    toml_dict["image_capture_rate"] = capture_rate
    toml_dict["maximum_period"] = duration
    toml_dict["active_cameras"] = [bool(camera_bit_flag & (1 << i)) for i in range(4)]

    with open(config_file_path, 'w') as f:
        toml.dump(toml_dict, f)

    if camera_params:
        main_config_path = os.path.join("config", "config.toml")
        main_config = toml.load(main_config_path)
        isp = main_config.setdefault("camera-isp", {})
        for key, value in camera_params.items():
            if value is not None:
                isp[key] = value
        with open(main_config_path, 'w') as f:
            toml.dump(main_config, f)
    
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
        print(f"Error: {path_out_file} file not created")
        return None
    
    dataset_path = path_out_file.read_text().strip()
    print(f"Test dataset generated at: {dataset_path}")
    return dataset_path

def run_dataset_processing(dataset_path, level_processing, rc_version, ld_version, bypass_prefilter_rejection=False):
    """
    This will call the binary to perform the dataset processing
    it will generate a processing.json that will be send to the mainboard

    here we do not need to write the toml file because they are read arguments
    
    TODO: it should have a timeout as well 
    """

    run_path = "."
    bin_name = "./bin/reprocess_dataset"
    
    dataset_folder = _dataset_folder_path(dataset_path)

    print(f"Running dataset processing on {dataset_folder}")
    print(f"Level of processing: {level_processing}")
    print(f"RC version: {rc_version}")
    print(f"LD version: {ld_version}")
    
    cmd = [bin_name, dataset_folder,
           "--target-stage", str(level_processing),
           "--overwrite",
           "--rc-version", str(rc_version),
           "--ld-version", str(ld_version)]
    if bypass_prefilter_rejection:
        cmd.append("--bypass-prefilter-rejection")

    result = subprocess.run(cmd,
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
    
    if return_code != 0:
        return None

    json_path = os.path.join(dataset_folder, "processing.json")
    
    # for now we will just return the same path
    return json_path


def package_od_csv_downlink(results_dir):
    """
    Create a zip containing only CSV products from an OD results directory.

    The od_result.json is intentionally handled separately and is not included
    in this archive.
    """

    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: OD results directory does not exist: {results_path}")
        return None

    csv_paths = sorted(path for path in results_path.glob("*.csv") if path.is_file())
    if not csv_paths:
        print(f"Warning: no CSV files found to package in {results_path}")
        return None

    timestamp = results_path.name
    zip_path = results_path / f"od_{timestamp}.zip"

    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for csv_path in csv_paths:
                zf.write(csv_path, arcname=csv_path.name)
    except Exception as e:
        print(f"Error creating OD CSV downlink zip: {e}")
        return None

    print(f"OD CSV downlink zip generated at: {zip_path}")
    return str(zip_path)


def run_orbit_determination(dataset_path, max_iter, max_runtime):
    """
    This will call the binary to perform the orbit determination
    it will generate a results.json that will be send to the mainboard

    for now this will just be a placeholder that will return the same path

    here we do not need to write the toml file because they are read arguments
    """
    
    dataset_folder = _dataset_folder_path(dataset_path)
    timeout = max_runtime + 20
    run_path = "."
    bin_name = "./bin/RUN_OD_ON_DATASET"
    
    
    try:
        result = subprocess.run([bin_name, dataset_folder,
                                 "--max-iterations", str(max_iter), 
                                 "--max-run-time", str(max_runtime)],
            cwd=run_path,
            # capture_output=True,
            timeout=timeout,
            text=True
        )
    except subprocess.TimeoutExpired:
        print(f"Orbit determination timed out after {timeout} seconds")
        return None
    
    try:
        return_code = result.returncode
        print(f"Return code: {return_code}")
    except Exception as e:
        print(f"Error capturing return code: {e}")
        return None

    if return_code != 0:
        return None
    
    # it is writing to path.out, but  I am actually not going to use it
    path_out_file = Path("path.out")
    if not path_out_file.exists():
        print("Error: path.out file not created")
        return None
    
    od_result_path = path_out_file.read_text().strip()
    print(f"Test dataset generated at: {od_result_path}")
    json_path = os.path.join(od_result_path, "od_result.json")
    package_od_csv_downlink(od_result_path)
    
    # for now we will just return the same path
    return json_path
