import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from camera_driver import JetsonCamera
from thread_shared import PayloadState, experiment_queue, log, state_manager, tx_queue
from file_downlink_manager import FileDownlinkManager

from external_bin_calls import is_dataset_running, run_inference, start_dataset, get_dataset_folder

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
        # get parameters like in run_inference
        camera_bit_flag = request.get("camera_bit_flag", 0)
        level_processing = request.get("level_processing", 0)
        width = request.get("width", 4608)
        height = request.get("height", 2592)
        downscale_factor = request.get("downscale_factor", 2.0)
        camera_defaults_selector = request.get("camera_defaults_selector", self.USE_PROGRAM_CAMERA_DEFAULTS)
        
        
        camera_params = None
        if camera_defaults_selector != self.USE_PROGRAM_CAMERA_DEFAULTS:
            camera_params = self._build_camera_params_from_request(request, camera_defaults_selector)
        
        max_period = float(request.get("max_period", 10.0))
        target_frame_nb = int(request.get("target_frame_nb", 4))
        capture_mode = str(request.get("capture_mode", "PERIODIC"))
        image_capture_rate = int(request.get("image_capture_rate", 1))
        imu_sample_rate_hz = float(request.get("imu_sample_rate_hz", 1.0))
        proc_stage = ExperimentThread._level_processing_to_proc_stage(level_processing)

        args = [
            "-m", str(max_period),
            "-n", str(target_frame_nb),
            "-M", capture_mode,
            "-r", str(image_capture_rate),
            "-s", str(imu_sample_rate_hz),
            "-p", str(proc_stage),
        ]

        # call start_dataset and get dataset folder
        ok, dataset_folder = start_dataset(args)
        if not ok:
            log.error("Dataset already running, experiment rejected")
            state_manager.set(PayloadState.FAIL)
            return

        state_manager.set(PayloadState.CAPTURING)

        file_manifest = self._init_file_manifest()

        if level_processing & self.PROCESS_DOWNSCALE_BIT:
            state_manager.set(PayloadState.DOWNSCALING)
            self._watch_and_downscale(dataset_folder, downscale_factor, file_manifest)
        else:
            while is_dataset_running():
                time.sleep(0.5)

        log.info("Dataset collection finished.")

        final_file_list = self._files_for_downlink(file_manifest)
        if not final_file_list:
            final_file_list = file_manifest["raw"]

        state_manager.set(PayloadState.DOWNLOAD)
        self.send_results(final_file_list)
        
    def experiment(
        self,
        camera_bit_flag: int,
        level_processing: int = 0,
        width: int = 4608,
        height: int = 2592,
        downscale_factor: float = 2.0,
        camera_params: dict | None = None,
    ):
        experiment_dir = self._create_experiment_output_dir()
        log.info("Experiment output directory: %s", experiment_dir)
        file_manifest = self._init_file_manifest()
        
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
        self._capture_from_cameras(
            cameras,
            experiment_dir,
            file_manifest,
        )
        log.info("Captured %d images", len(file_manifest.get("raw", [])))
        log.debug("After capture: %s", file_manifest)
        self._close_cameras(cameras)

        # 3. run the processing
        # check to see if should downscale image
        if level_processing & self.PROCESS_DOWNSCALE_BIT:
            state_manager.set(PayloadState.DOWNSCALING)
            self._downscale_images(
                downscale_factor,
                experiment_dir,
                file_manifest,
            )
        log.debug("After downscaling: %s", file_manifest)
            
        # check to see if should run prefiltering
        if level_processing & self.PROCESS_PREFILTER_BIT:
            state_manager.set(PayloadState.PREFILTERING)
            self._prefilter_images(
                file_manifest,
            )
        log.debug("After prefiltering: %s", file_manifest)

        # check to see if should run inference
        if level_processing & self.PROCESS_INFERENCE_BIT:
            state_manager.set(PayloadState.INFERENCE)
            self._run_inference(
                experiment_dir,
                file_manifest,
            )
        log.debug("After inference: %s", file_manifest)
            
        # send the finished experiment command
        # TODO - not sure that I want to do this here
        command = Command("EXPERIMENT_FINISHED")
        tx_queue.put(pack(command))
        log.info("Sending EXPERIMENT_FINISHED command (experiment finished)")
        
        # get the files that we want to send to the mainboard
        # for now it will be the files in the results and downscale folder inside the experiment folder
        # we need to create a list with the paths for all those files
        final_file_list = self._files_for_downlink(file_manifest)
        if not final_file_list:
            log.warning(
                "No files found in file_manifest downscaled/results for %s. "
                "Falling back to raw/current_image_path_list.",
                experiment_dir,
            )
            final_file_list = file_manifest["raw"] 
        log.info("Prepared %d files for downlink", len(final_file_list))
        
        
        # finished experiment, want to go to downlink mode
        state_manager.set(PayloadState.DOWNLOAD)
        self.send_results(final_file_list)
        
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

        # wait 2 seconds for camera to stabilize
        time.sleep(2)
        
        return cameras

    def _capture_from_cameras(
        self,
        cameras: dict[int, JetsonCamera],
        experiment_dir: Path,
        file_manifest: dict,
    ) -> list[str]:
        if not cameras:
            return []

        output_dir = experiment_dir / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        saved_paths: list[str] = []

        for sensor_id in sorted(cameras.keys()):
            camera = cameras[sensor_id]
            log.info("Capturing image from camera sensor-id=%d", sensor_id)
            frame = camera.capture()
            output_path = output_dir / f"{sensor_id}_{timestamp}.jpeg"
            camera.save(frame, str(output_path))
            saved_paths.append(str(output_path))

        file_manifest["raw"].extend(saved_paths)


    def _close_cameras(self, cameras: dict[int, JetsonCamera]) -> None:
        for sensor_id, camera in cameras.items():
            try:
                camera.close()
            except Exception as exc:
                log.warning("Failed to close camera sensor-id=%d: %s", sensor_id, exc)



    def _downscale_images(
        self,
        downscale_factor: float = 2.0,
        experiment_dir: Path | None = None,
        file_manifest: dict | None = None,
    ) -> list[str]:
        """
        Using the images in file_manifest raw. For each of the images, it will perform the necessary downscale to the image
        it will save the new image in a new path and add it to the manifest
        """
        output_dir = experiment_dir / "downscaled"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if downscale_factor <= 0:
            log.warning(
                "Invalid downscale_factor=%s. Falling back to 2.0",
                downscale_factor,
            )
            downscale_factor = 2.0

        downscaled_paths: list[str] = []
        for image_path in file_manifest.get("raw", []):
            src = Path(image_path)
            img = cv2.imread(str(src))
            if img is None:
                raise RuntimeError(f"Failed to read image for downscaling: {src}")

            height, width = img.shape[:2]
            resized_width = max(1, int(round(width / downscale_factor)))
            resized_height = max(1, int(round(height / downscale_factor)))
            resized = cv2.resize(img, (resized_width, resized_height))
            dst = output_dir / f"{src.stem}.jpeg"
            ok = cv2.imwrite(str(dst), resized)
            if not ok:
                raise RuntimeError(f"Failed to write downscaled image: {dst}")
            downscaled_paths.append(str(dst))

        log.info(
            "Downscaled %d images with factor=%s",
            len(downscaled_paths),
            downscale_factor,
        )
        if file_manifest is not None:
            file_manifest["downscaled"].extend(downscaled_paths)


    def _prefilter_images(self, file_manifest: dict | None = None) -> list[str]:
        """
        Receives a list of image paths and will perform pre filtering
        It will return a new list of image paths contain only the images that passed pre filtering
        For each of the images path, it will create a json file containing the results of the prefiltering
        """
        
        log.error("Prefiltering stage not implemented")


    def _run_inference(
        self,
        experiment_dir: Path,
        file_manifest: dict | None = None,
    ) -> list[str]:
        """
        Receive a list with the file path to the images it should run inference on
        for each image in the image_paths, it will create a json with the results of the experiment
        """
        
        output_folder = experiment_dir / "results"
        output_folder.mkdir(parents=True, exist_ok=True)
        

        log.info("Running inference on images...")
        for image_path in file_manifest.get("raw", []):

            log.info(f"Running inference on {image_path} and {output_folder}")
            run_inference(str(image_path), f"{str(output_folder)}/")

        # for now will return whatever data was generated by the inferece
        # return list dir for the output folder

        result_paths = [str(p) for p in sorted(output_folder.rglob("*")) if p.is_file()]
        if file_manifest is not None:
            file_manifest["results"].extend(result_paths)
        return result_paths

    def _create_experiment_output_dir(self) -> Path:
        root = Path("experiments")
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base = root / f"experiment_{timestamp}"

        if not base.exists():
            base.mkdir(parents=True, exist_ok=False)
            return base

        suffix = 1
        while True:
            candidate = root / f"experiment_{timestamp}_{suffix:02d}"
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            suffix += 1

    @staticmethod
    def _init_file_manifest() -> dict:
        return {
            "raw": [],
            "downscaled": [],
            "results": [],
        }

    @staticmethod
    def _files_for_downlink(file_manifest: dict) -> list[str]:
        """
        Returns the files that should be sent to the mainboard. downscaled images and result
        returns empty list in case they are missing from the file manifest
        """
        return [*file_manifest.get("downscaled", []), *file_manifest.get("results", [])]
            
        

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

    def _watch_and_downscale(self, dataset_folder: Path, downscale_factor: float, file_manifest: dict) -> None:
        """
        Watches dataset_folder for new PNGs while the dataset subprocess is running
        and downscales each one once it is fully written.
        """
        seen: set[Path] = set()
        stable_candidates: dict[Path, int] = {}  
        
        while is_dataset_running() and not dataset_folder.exists():
            time.sleep(0.1)

        if not dataset_folder.exists():
            log.error("Dataset folder never appeared: %s", dataset_folder)
            return

        output_dir = dataset_folder / "downscaled"
        output_dir.mkdir(parents=True, exist_ok=True)

        while is_dataset_running():
            for png in dataset_folder.glob("*.png"):
                if png in seen:
                    continue

                current_size = png.stat().st_size
                if png in stable_candidates:
                    if stable_candidates[png] == current_size:
                        # size unchanged since last poll, can downscale now
                        self._downscale_single(png, output_dir, downscale_factor, file_manifest)
                        seen.add(png)
                        del stable_candidates[png]
                    else:
                        stable_candidates[png] = current_size
                else:
                    stable_candidates[png] = current_size

            time.sleep(0.5)

        # dataset finished
        for png in dataset_folder.glob("*.png"):
            if png in seen:
                continue
            self._downscale_single(png, output_dir, downscale_factor, file_manifest)
            seen.add(png)


    def _downscale_single(self, src: Path, output_dir: Path, downscale_factor: float, file_manifest: dict) -> None:
        img = cv2.imread(str(src))
        if img is None:
            log.error("Failed to read image for downscaling: %s", src)
            return

        h, w = img.shape[:2]
        resized = cv2.resize(img, (max(1, int(round(w / downscale_factor))), max(1, int(round(h / downscale_factor)))))
        dst = output_dir / f"{src.stem}.jpeg"
        if not cv2.imwrite(str(dst), resized):
            log.error("Failed to write downscaled image: %s", dst)
            return

        file_manifest["downscaled"].append(str(dst))
        file_manifest["raw"].append(str(src))
        log.info("Downscaled %s -> %s", src.name, dst.name)

    @staticmethod
    def _level_processing_to_proc_stage(level_processing: int) -> int:
        if level_processing & ExperimentThread.PROCESS_INFERENCE_BIT:
            return 3  # ProcessingStage::LDNeted
        if level_processing & ExperimentThread.PROCESS_PREFILTER_BIT:
            return 1  # ProcessingStage::Prefiltered
        return 0      # ProcessingStage::NotPrefiltered