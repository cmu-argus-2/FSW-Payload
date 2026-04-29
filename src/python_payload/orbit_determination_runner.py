"""
This is hte folder that will contain the logic for the orbit determination experiment
for now it will implement the dataset collection command
not sure if I will make a different file for the other od commands or not
"""

import threading
from file_downlink_manager import FileDownlinkManager
from external_bin_calls import run_dataset_collection
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
        imu_hz = args.get("imu_hz", 100)
        camera_hz = args.get("camera_hz", 10)
        duration = args.get("duration", 60)
        
        state_manager.set(PayloadState.CAPTURING)
        dataset_json_path = run_dataset_collection(imu_hz, camera_hz, duration)
        
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