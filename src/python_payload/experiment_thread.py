import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from camera_driver import JetsonCamera
from thread_shared import PayloadState, experiment_queue, log, state_manager, tx_queue
from file_downlink_manager import FileDownlinkManager

from splat.splat.telemetry_codec import Command, pack

import math
import os


class ExperimentThread(threading.Thread):
    """Consumes experiment requests and runs them asynchronously."""

    PROCESS_DOWNSCALE_BIT = 1 << 0
    PROCESS_PREFILTER_BIT = 1 << 1
    PROCESS_INFERENCE_BIT = 1 << 2
    USE_PROGRAM_CAMERA_DEFAULTS = -1

    def __init__(self, stop_event: threading.Event):
        super().__init__(name="ExperimentThread", daemon=True)
        self.stop_event = stop_event
        self.downlink_manager = FileDownlinkManager(stop_event=stop_event)

    def run(self):
        log.info("Experiment thread started")
        while not self.stop_event.is_set():
            try:
                request = experiment_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._run_experiment(request)

    def _run_experiment(self, request: dict):
        ts = int(request.get("ts", 0))
        camera_bit_flag = int(request.get("camera_bit_flag", 0))
        level_processing = int(request.get("level_processing", 0))
        width = int(request.get("width", 4608))
        height = int(request.get("height", 2592))
        camera_defaults_selector = int(
            request.get("camera_defaults_selector", self.USE_PROGRAM_CAMERA_DEFAULTS)
        )
        camera_params = self._build_camera_params_from_request(
            request,
            camera_defaults_selector,
        )

        log.info(
            (
                "Starting experiment ts=%s camera_mask=%s level=%s width=%s height=%s "
                "camera_defaults_selector=%s"
            ),
            ts,
            camera_bit_flag,
            level_processing,
            width,
            height,
            camera_defaults_selector,
        )

        try:
            self.experiment(
                camera_bit_flag,
                level_processing=level_processing,
                width=width,
                height=height,
                camera_params=camera_params,
            )
            log.info("Experiment completed")
        except Exception as exc:
            log.error("Experiment failed: %s", exc)
            state_manager.set(PayloadState.FAIL)

    def experiment(
        self,
        camera_bit_flag: int,
        level_processing: int = 0,
        width: int = 4608,
        height: int = 2592,
        camera_params: dict | None = None,
    ):
        
        # 1. turn on the required cameras
        state_manager.set(PayloadState.TURNING_ON)
        cameras = self._start_cameras(
            camera_bit_flag,
            width=width,
            height=height,
            camera_params=camera_params,
        )

        if not cameras:
            log.info("No cameras selected; experiment finished without capture")
            state_manager.set(PayloadState.IDLE)
            return []

        # 2. capture and save images from the required cameras
        state_manager.set(PayloadState.CAPTURING)
        current_image_path_list = self._capture_from_cameras(cameras)
        log.info("Captured %d images", len(current_image_path_list))
        self._close_cameras(cameras)

        # 3. run the processing
        # check to see if should downscale image
        if level_processing & self.PROCESS_DOWNSCALE_BIT:
            state_manager.set(PayloadState.DOWNSCALING)
            current_image_path_list = self._downscale_images(current_image_path_list)
            
        # check to see if should run prefiltering
        if level_processing & self.PROCESS_PREFILTER_BIT:
            state_manager.set(PayloadState.PREFILTERING)
            current_image_path_list = self._prefilter_images(current_image_path_list)

        # check to see if should run inference
        if level_processing & self.PROCESS_INFERENCE_BIT:
            state_manager.set(PayloadState.INFERENCE)
            current_image_path_list = self._run_inference(current_image_path_list)
            
        # send the finished experiment command
        # TODO - not sure that I want to do this here
        command = Command("EXPERIMENT_FINISHED")   # TODO - change this to proper finish experiment command
        tx_queue.put(pack(command))
        log.info("Sending EXPERIMENT_FINISHED command (experiment finished)")
        
        # finished experiment, want to go to downlink mode
        state_manager.set(PayloadState.DOWNLOAD)
        self.send_results(current_image_path_list)  # TODO currently only sending the images, need to change later on to send the results as well
        
    def send_results(self, final_file_list: list[str]):
        """
        This will be the last part of the experiment sending all the files to the mainboard
        will send all the files that are on the experiment file_list
        """
        ok = self.downlink_manager.send_files(final_file_list, stop_on_failure=True)
        if not ok:
            log.error("Downlink sequence failed")
            state_manager.set(PayloadState.FAIL)
        else:
            log.info("Downlink sequence completed for %s files", len(final_file_list))
        

    def send_file(self, file_path: str):
        """
        Compatibility wrapper around the new downlink manager.
        """
        result = self.downlink_manager.send_file(file_path)
        if not result.success:
            log.error("Downlink failed for %s: %s", file_path, result.reason)
            return False
        log.info("Downlink completed for %s", file_path)
        return True

    def _start_cameras(
        self,
        camera_bit_flag: int,
        width: int = 4608,
        height: int = 2592,
        camera_params: dict | None = None,
    ) -> dict[int, JetsonCamera]:
        width, height = self._validate_dimensions(width, height)
        active_sensor_ids = [sensor_id for sensor_id in range(4) if camera_bit_flag & (1 << sensor_id)]
        if not active_sensor_ids:
            return {}

        log.info(
            "Using capture dimensions %dx%d",
            width,
            height,
        )

        cameras: dict[int, JetsonCamera] = {}
        camera_params = camera_params or {}
        for sensor_id in active_sensor_ids:
            log.info("Initializing camera sensor-id=%d", sensor_id)
            camera = JetsonCamera(
                sensor_id=sensor_id,
                width=width,
                height=height,
                **camera_params,
            )
            camera.open()
            cameras[sensor_id] = camera

        # wait 5 seconds for camera to stabilize
        #time.sleep(5)  # no need to sleep while experimnet with things
        
        return cameras

    def _capture_from_cameras(self, cameras: dict[int, JetsonCamera]) -> list[str]:
        if not cameras:
            return []

        output_dir = Path("images_raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        saved_paths: list[str] = []

        for sensor_id in sorted(cameras.keys()):
            camera = cameras[sensor_id]
            log.info("Capturing image from camera sensor-id=%d", sensor_id)
            frame = camera.capture()
            output_path = output_dir / f"raw_{sensor_id}_{timestamp}.jpeg"
            camera.save(frame, str(output_path))
            saved_paths.append(str(output_path))

        return saved_paths

    def _close_cameras(self, cameras: dict[int, JetsonCamera]) -> None:
        for sensor_id, camera in cameras.items():
            try:
                camera.close()
            except Exception as exc:
                log.warning("Failed to close camera sensor-id=%d: %s", sensor_id, exc)



    def _downscale_images(self, image_paths: list[str]) -> list[str]:
        """
        Will receive a list of image paths. For each of the images, it will perform the necessary downsample to the image
        It will prepare the image to feed to the model
        it will save the new image in a new path and return a new image_path list with the new paths
        """
        output_dir = Path("images_downscaled")
        output_dir.mkdir(parents=True, exist_ok=True)

        downscaled_paths: list[str] = []
        for image_path in image_paths:
            src = Path(image_path)
            img = cv2.imread(str(src))
            if img is None:
                raise RuntimeError(f"Failed to read image for downscaling: {src}")

            height, width = img.shape[:2]
            resized = cv2.resize(img, (max(1, width // 2), max(1, height // 2)))
            dst = output_dir / f"downscaled_{src.stem}.jpeg"
            ok = cv2.imwrite(str(dst), resized)
            if not ok:
                raise RuntimeError(f"Failed to write downscaled image: {dst}")
            downscaled_paths.append(str(dst))

        log.info("Downscaled %d images", len(downscaled_paths))
        return downscaled_paths

    def _prefilter_images(self, image_paths: list[str]) -> list[str]:
        """
        Receives a list of image paths and will perform pre filtering
        It will return a new list of image paths contain only the images that passed pre filtering
        For each of the images path, it will create a json file containing the results of the prefiltering
        """
        
        log.error("Prefiltering stage not implemented")
        return image_paths

    def _run_inference(self, image_paths: list[str]) -> None:
        """
        Receive a list with the file path to the images it should run inference on
        for each image in the image_paths, it will create a json with the results of the experiment
        """
        log.error("Inference stage not implemented")
        

    def _validate_dimensions(self, width: int, height: int) -> tuple[int, int]:
        if width <= 0 or height <= 0:
            log.warning(
                "Invalid dimensions width=%s height=%s. Falling back to 4608x2592",
                width,
                height,
            )
            return (4608, 2592)
        return (width, height)

    def _build_camera_params_from_request(
        self,
        request: dict,
        camera_defaults_selector: int,
    ) -> dict:
        if camera_defaults_selector == self.USE_PROGRAM_CAMERA_DEFAULTS:
            return {}

        exposuretimerange = self._decode_int_range(
            request.get("exposuretimerange_low", 0),
            request.get("exposuretimerange_high", 0),
        )
        gainrange = self._decode_float_range(
            request.get("gainrange_low", 0.0),
            request.get("gainrange_high", 0.0),
        )
        ispdigitalgainrange = self._decode_float_range(
            request.get("ispdigitalgainrange_low", 0.0),
            request.get("ispdigitalgainrange_high", 0.0),
        )

        return {
            "fps": int(request.get("fps", 14)),
            "wbmode": int(request.get("wbmode", 0)),
            "aelock": self._to_bool(request.get("aelock", 0)),
            "awblock": self._to_bool(request.get("awblock", 0)),
            "exposuretimerange": exposuretimerange,
            "gainrange": gainrange,
            "ispdigitalgainrange": ispdigitalgainrange,
            "ee_mode": int(request.get("ee_mode", 1)),
            "ee_strength": float(request.get("ee_strength", -1.0)),
            "aeantibanding": int(request.get("aeantibanding", 1)),
            "exposurecompensation": float(request.get("exposurecompensation", 0.0)),
            "tnr_mode": int(request.get("tnr_mode", 1)),
            "tnr_strength": float(request.get("tnr_strength", -1.0)),
            "saturation": float(request.get("saturation", 1.0)),
        }

    @staticmethod
    def _to_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        return bool(int(value))

    @staticmethod
    def _decode_int_range(low, high) -> tuple[int, int] | None:
        low_i = int(low)
        high_i = int(high)
        if low_i == 0 and high_i == 0:
            return None
        return (low_i, high_i)

    @staticmethod
    def _decode_float_range(low, high) -> tuple[float, float] | None:
        low_f = float(low)
        high_f = float(high)
        if low_f == 0.0 and high_f == 0.0:
            return None
        return (low_f, high_f)

