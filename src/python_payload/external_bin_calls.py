"""
This file will implement the code that will call the c++ bins to perform the actions
like record a dataset, or run inference...
"""

import os
import subprocess
import sys
import tempfile
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

    Parameters are passed to the binary via temporary config files so the committed
    dataset_config.toml and config.toml are never modified.
    """

    timeout = duration + 20
    run_path = "."
    bin_name = "./bin/run_dataset"
    path_out_file = Path("path.out")
    path_out_file.unlink(missing_ok=True)

    ds_config_tmp = None
    system_config_tmp = None
    try:
        ds_toml = toml.load(os.path.join("config", "dataset_config.toml"))
        ds_toml["imu_sample_rate_hz"] = imu_hz
        ds_toml["image_capture_rate"] = capture_rate
        ds_toml["maximum_period"] = duration
        ds_toml["active_cameras"] = [bool(camera_bit_flag & (1 << i)) for i in range(4)]

        with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
            toml.dump(ds_toml, f)
            ds_config_tmp = f.name

        cmd = [bin_name, "--config", ds_config_tmp]

        if camera_params:
            main_config = toml.load(os.path.join("config", "config.toml"))
            isp = main_config.setdefault("camera-isp", {})
            for key, value in camera_params.items():
                if value is not None:
                    isp[key] = value
            with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
                toml.dump(main_config, f)
                system_config_tmp = f.name
            cmd += ["--system-config", system_config_tmp]

        try:
            result = subprocess.run(cmd,
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

        # If the result failed to write the output, don't read it, run_dataset only writes output if successful
        if return_code != 0:
            return None

        if not path_out_file.exists():
            print(f"Error: {path_out_file} file not created")
            return None

        dataset_path = path_out_file.read_text().strip()
        if not dataset_path:
            print(f"Error: {path_out_file} file is empty")
            return None

        print(f"Test dataset generated at: {dataset_path}")
        return dataset_path

    finally:
        if ds_config_tmp:
            Path(ds_config_tmp).unlink(missing_ok=True)
        if system_config_tmp:
            Path(system_config_tmp).unlink(missing_ok=True)

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


def _collect_od_run_result(succeeded: bool):
    path_out_file = Path("path.out")
    if not path_out_file.exists():
        print("Error: path.out file not created")
        return None, succeeded

    od_result_path = path_out_file.read_text().strip()
    if not od_result_path:
        print("Error: path.out file is empty")
        return None, succeeded

    print(f"OD results directory generated at: {od_result_path}")
    json_path = os.path.join(od_result_path, "od_result.json")
    if not Path(json_path).exists():
        print(f"Error: OD result JSON not found: {json_path}")
        json_path = None

    return json_path, succeeded


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

    Path("path.out").unlink(missing_ok=True)

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
        return _collect_od_run_result(succeeded=False)
    
    try:
        return_code = result.returncode
        print(f"Return code: {return_code}")
    except Exception as e:
        print(f"Error capturing return code: {e}")
        return None, False

    return _collect_od_run_result(succeeded=(return_code == 0))