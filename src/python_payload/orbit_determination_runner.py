"""
This is the folder that will contain the logic for the orbit determination experiment
for now it will implement the dataset collection command
not sure if I will make a different file for the other od commands or not
"""

import json
import threading
from pathlib import Path

import cv2

from file_downlink_manager import FileDownlinkManager
from external_bin_calls import run_dataset_processing, run_orbit_determination
from splat.splat.telemetry_codec import Command, pack
from thread_shared import (
    PayloadState,
    log,
    state_manager,
    tx_queue,
)
            
class DatasetProcessingRunner:
    
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event
        self.downlink_manager = FileDownlinkManager(stop_event=stop_event)

    def run(self, args: dict):
        log.info("Running dataset processing with args: %s", args)
        state_manager.set(PayloadState.TURNING_ON)

        dataset_json_path = args.get("string_command", "").rstrip("\x00")
        level_processing = args.get("level_processing", 1)
        rc_version = args.get("rc_version", 2)
        ld_version = args.get("ld_version", 3)
        bypass_prefilter_rejection = args.get("bypass_preflt_rej", True)

        state_manager.set(PayloadState.CAPTURING)
        processing_json_path = run_dataset_processing(dataset_json_path,
                                                      level_processing, rc_version, ld_version,
                                                      bypass_prefilter_rejection)

        if processing_json_path is None or processing_json_path == -1:
            log.error("Dataset processing failed")
            state_manager.set(PayloadState.FAIL)
            return

        log.info("Dataset processing completed, processing path: %s", processing_json_path)

        dataset_folder = Path(processing_json_path).parent
        thumb_pairs = self._generate_thumbnails(dataset_folder, processing_json_path)
        thumb_paths = [p for p, _ in thumb_pairs]

        command = Command("EXPERIMENT_FINISHED")
        tx_queue.put(pack(command))
        log.info("Sending EXPERIMENT_FINISHED command (experiment finished)")

        state_manager.set(PayloadState.DOWNLOAD)
        downlink_list = [processing_json_path] + thumb_paths
        ok = self.downlink_manager.send_files(downlink_list)
        if not ok:
            log.error("Downlink sequence failed")
            state_manager.set(PayloadState.FAIL)
        else:
            log.info("Downlink sequence completed for %s files", len(downlink_list))
            if thumb_pairs:
                self._mark_thumbnails_sent(dataset_folder, [fn for _, fn in thumb_pairs])

    def _generate_thumbnails(self, dataset_folder: Path, processing_json_path: str) -> list[tuple[str, str]]:
        """
        Generate thumbnails for the top-N unsent ranked frames and return (thumb_path, raw_filename) pairs.
        Reads downlink_config.json for N and target dimensions. Returns [] if config absent or N==0.
        Skips any thumbnail that exceeds 100 KB after writing.
        """
        config_path = dataset_folder / "downlink_config.json"
        if not config_path.exists():
            return []

        try:
            with open(config_path) as f:
                config = json.load(f)
            n_thumbnails = int(config.get("n_thumbnails", 0))
            thumb_width = int(config.get("thumb_width", 640))
            thumb_height = int(config.get("thumb_height", 480))
        except Exception as exc:
            log.error("Failed to read downlink_config.json: %s", exc)
            return []

        if n_thumbnails <= 0:
            return []

        try:
            with open(processing_json_path) as f:
                processing = json.load(f)
        except Exception as exc:
            log.error("Failed to read processing.json for thumbnail selection: %s", exc)
            return []

        frames = processing.get("processed_frames", [])
        frames_sorted = sorted(
            frames,
            key=lambda fr: (fr.get("annotation_state", 0), fr.get("rank", 0.0), fr.get("timestamp", 0)),
            reverse=True,
        )

        # Load marker of already-sent raw image filenames
        marker_path = dataset_folder / "downlinked_thumbnails.json"
        sent_set: set[str] = set()
        if marker_path.exists():
            try:
                with open(marker_path) as f:
                    sent_set = set(json.load(f).get("sent", []))
            except Exception as exc:
                log.warning("Failed to read downlinked_thumbnails.json: %s", exc)

        downlink_dir = dataset_folder / "downlink"
        downlink_dir.mkdir(parents=True, exist_ok=True)

        results: list[tuple[str, str]] = []
        for frame in frames_sorted:
            if len(results) >= n_thumbnails:
                break

            raw_path = frame.get("raw_image", {}).get("path", "")
            if not raw_path:
                continue

            raw_filename = Path(raw_path).name
            if raw_filename in sent_set:
                continue

            img = cv2.imread(raw_path)
            if img is None:
                log.warning("Could not read image for thumbnail: %s", raw_path)
                continue

            timestamp = frame.get("timestamp", 0)
            cam_id = frame.get("cam_id", 0)
            thumb_path = downlink_dir / f"thumb_{timestamp}_{cam_id}.jpg"

            resized = cv2.resize(img, (thumb_width, thumb_height))

            accepted = False
            size = 0
            for quality in (100, 95, 85, 60):
                if not cv2.imwrite(str(thumb_path), resized, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                    log.warning("Failed to write thumbnail at quality=%d: %s", quality, thumb_path)
                    break
                size = thumb_path.stat().st_size
                if size <= 250_000:
                    accepted = True
                    break
                log.debug("Thumbnail %s at quality=%d is %d bytes, retrying", thumb_path.name, quality, size)

            if not accepted:
                log.warning("Thumbnail %s exceeds 250 KB at all quality levels (%d bytes), skipping", thumb_path.name, size)
                if thumb_path.exists():
                    thumb_path.unlink()
                continue

            log.info("Generated thumbnail %s (%d bytes)", thumb_path.name, size)
            results.append((str(thumb_path), raw_filename))

        return results

    def _mark_thumbnails_sent(self, dataset_folder: Path, raw_filenames: list[str]) -> None:
        """Append raw_filenames to downlinked_thumbnails.json in the dataset folder."""
        marker_path = dataset_folder / "downlinked_thumbnails.json"
        existing: set[str] = set()
        if marker_path.exists():
            try:
                with open(marker_path) as f:
                    existing = set(json.load(f).get("sent", []))
            except Exception as exc:
                log.warning("Failed to read downlinked_thumbnails.json for update: %s", exc)
        existing.update(raw_filenames)
        try:
            with open(marker_path, "w") as f:
                json.dump({"sent": sorted(existing)}, f)
            log.info("Updated downlinked_thumbnails.json with %s entries", len(raw_filenames))
        except Exception as exc:
            log.error("Failed to write downlinked_thumbnails.json: %s", exc)
            

class DatasetODRunner:
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event
        self.downlink_manager = FileDownlinkManager(stop_event=stop_event)

    @staticmethod
    def _find_partial_od_result() -> str | None:
        """Return od_result.json path from path.out if it exists, else None."""
        try:
            path_out = Path("path.out")
            if not path_out.exists():
                log.warning("_find_partial_od_result: path.out does not exist — binary did not write output path on failure")
                return None
            od_result_dir = path_out.read_text().strip()
            log.info("_find_partial_od_result: path.out contents: %r", od_result_dir)
            if not od_result_dir:
                log.warning("_find_partial_od_result: path.out is empty")
                return None
            candidate = Path(od_result_dir) / "od_result.json"
            if not candidate.exists():
                log.warning("_find_partial_od_result: od_result.json not found at %s", candidate)
                return None
            log.info("_find_partial_od_result: found partial result at %s", candidate)
            return str(candidate)
        except Exception as exc:
            log.error("_find_partial_od_result: unexpected error: %s", exc)
            return None

    def run(self, args: dict):
        
        log.info("Running dataset orbit determination with args: %s", args)
        state_manager.set(PayloadState.TURNING_ON)
        
        # call the binary to perform the dataset processing
        dataset_json_path = args.get("string_command", "").rstrip("\x00")  # remove trailing 0s
        max_iter = args.get("max_iteration", 1000)
        max_runtime = args.get("max_runtime", 300)
        
        state_manager.set(PayloadState.CAPTURING)
        results_json_path, od_succeeded = run_orbit_determination(
            dataset_json_path,
            max_iter,
            max_runtime,
        )

        od_succeeded = od_succeeded and results_json_path is not None and results_json_path != -1

        if results_json_path is None or results_json_path == -1:
            log.error("Orbit determination failed")
            state_manager.set(PayloadState.FAIL)
            results_json_path = self._find_partial_od_result()
            if results_json_path is None:
                log.error("No OD results to downlink, sending EXPERIMENT_FINISHED")
                command = Command("EXPERIMENT_FINISHED")
                tx_queue.put(pack(command))
                return
            log.warning("Partial od_result.json found, will attempt downlink: %s", results_json_path)

        log.info("Dataset OD results json path: %s", results_json_path)
        command = Command("EXPERIMENT_FINISHED")
        tx_queue.put(pack(command))
        log.info("Sending EXPERIMENT_FINISHED command (experiment finished)")

        state_manager.set(PayloadState.DOWNLOAD)

        downlink_list = [results_json_path]
        if od_succeeded:
            state_estimates_path = Path(results_json_path).parent / "state_estimates.csv"
            if state_estimates_path.exists():
                downlink_list.append(str(state_estimates_path))
            else:
                log.warning("state_estimates.csv not found in results folder, skipping")

        ok = self.downlink_manager.send_files(downlink_list)
        if not ok:
            log.error("Downlink sequence failed")
            state_manager.set(PayloadState.FAIL)
        elif not od_succeeded:
            log.info("Partial OD result downlinked (%s files)", len(downlink_list))
            state_manager.set(PayloadState.FAIL)
        else:
            log.info("Downlink sequence completed for %s files", len(downlink_list))
