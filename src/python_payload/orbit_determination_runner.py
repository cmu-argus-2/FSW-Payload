"""
This is hte folder that will contain the logic for the orbit determination experiment
for now it will implement the dataset collection command
not sure if I will make a different file for the other od commands or not
"""

import threading
from file_downlink_manager import FileDownlinkManager
from experiment_runner import ExperimentRunner
from external_bin_calls import run_dataset_collection, run_dataset_processing, run_orbit_determination
from splat.splat.telemetry_codec import Command, pack
from thread_shared import (
    PayloadState,
    log,
    set_inference_return_code,
    state_manager,
    tx_queue,
)


class DatasetCollectionRunner:
    
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event
        self.downlink_manager = FileDownlinkManager(stop_event=stop_event)

    def run(self, args: dict):
        
        log.info("Running dataset collection with args: %s", args)
        state_manager.set(PayloadState.TURNING_ON)
        
        # call the binary to perform the dataset collection
        camera_bit_flag = args.get("camera_bit_flag", 0b0001)
        imu_hz = args.get("imu_hz", 100)
        capture_rate = args.get("capture_rate", 10)
        duration = args.get("duration", 60)
        camera_defaults_selector = args.get("camera_defaults_selector", ExperimentRunner.USE_PROGRAM_CAMERA_DEFAULTS)
        camera_params = ExperimentRunner._build_camera_params_from_request(args, camera_defaults_selector)
        
        state_manager.set(PayloadState.CAPTURING)
        dataset_json_path = run_dataset_collection(camera_bit_flag, capture_rate, imu_hz, 
                                                   duration, camera_params)
        
        # check teh return value
        if dataset_json_path is None or dataset_json_path == -1:
            log.error("Dataset collection failed")
            state_manager.set(PayloadState.FAIL)
            return
        
        # Send the command finished command
        log.info("Dataset collection completed, dataset path: %s", dataset_json_path)
        command = Command("EXPERIMENT_FINISHED")
        tx_queue.put(pack(command))
        log.info("Sending EXPERIMENT_FINISHED command (experiment finished)")
        
        # send the files to the mainboard for downlink
        state_manager.set(PayloadState.DOWNLOAD)
        
        ok = self.downlink_manager.send_files([dataset_json_path])
        if not ok:
            log.error("Downlink sequence failed")
            state_manager.set(PayloadState.FAIL)
        else:
            log.info(f"Downlink sequence completed for {dataset_json_path}")
            
            
class DatasetProcessingRunner:
    
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event
        self.downlink_manager = FileDownlinkManager(stop_event=stop_event)

    def run(self, args: dict):
        
        log.info("Running dataset processing with args: %s", args)
        state_manager.set(PayloadState.TURNING_ON)
        
        # call the binary to perform the dataset processing
        dataset_json_path = args.get("string_command", "").rstrip("\x00")  # remove trailing 0s
        level_processing = args.get("level_processing", 1)
        rc_version = args.get("rc_version", 5)
        ld_version = args.get("ld_version", 3)
        bypass_prefilter_rejection = args.get("bypass_preflt_rej", True)
        
        state_manager.set(PayloadState.CAPTURING)
        dataset_json_path = run_dataset_processing(dataset_json_path,
                                                   level_processing, rc_version, ld_version,
                                                   bypass_prefilter_rejection)
        
        # check teh return value
        if dataset_json_path is None or dataset_json_path == -1:
            log.error("Dataset processing failed")
            state_manager.set(PayloadState.FAIL)
            return
        
        # Send the command finished command
        log.info("Dataset processing completed, dataset path: %s", dataset_json_path)
        command = Command("EXPERIMENT_FINISHED")
        tx_queue.put(pack(command))
        log.info("Sending EXPERIMENT_FINISHED command (experiment finished)")
        
        # send the files to the mainboard for downlink
        state_manager.set(PayloadState.DOWNLOAD)
        
        ok = self.downlink_manager.send_files([dataset_json_path])
        if not ok:
            log.error("Downlink sequence failed")
            state_manager.set(PayloadState.FAIL)
        else:
            log.info(f"Downlink sequence completed for {dataset_json_path}")
            

class DatasetODRunner:
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event
        self.downlink_manager = FileDownlinkManager(stop_event=stop_event)

    def run(self, args: dict):
        
        log.info("Running dataset orbit determination with args: %s", args)
        state_manager.set(PayloadState.TURNING_ON)
        
        # call the binary to perform the dataset processing
        dataset_json_path = args.get("string_command", "").rstrip("\x00")  # remove trailing 0s
        max_iter = args.get("max_iterations", 1)
        max_runtime = args.get("max_runtime", 1)
        
        state_manager.set(PayloadState.CAPTURING)
        results_json_path = run_orbit_determination(dataset_json_path, max_iter, max_runtime)
        
        # check teh return value
        if results_json_path is None or results_json_path == -1:
            log.error("Dataset processing failed")
            state_manager.set(PayloadState.FAIL)
            return
        
        # Send the command finished command
        log.info("Dataset processing completed, results json path: %s", results_json_path)
        command = Command("EXPERIMENT_FINISHED")
        tx_queue.put(pack(command))
        log.info("Sending EXPERIMENT_FINISHED command (experiment finished)")
        
        # send the files to the mainboard for downlink
        state_manager.set(PayloadState.DOWNLOAD)
        
        ok = self.downlink_manager.send_files([results_json_path])
        if not ok:
            log.error("Downlink sequence failed")
            state_manager.set(PayloadState.FAIL)
        else:
            log.info(f"Downlink sequence completed for {results_json_path}")
