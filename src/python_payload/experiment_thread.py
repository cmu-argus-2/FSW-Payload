import queue
import threading
import time
from experiment_runner import ExperimentRunner
from orbit_determination_runner import DatasetProcessingRunner, DatasetODRunner #,  DatasetCollectionRunner,
from thread_shared import (
    experiment_queue,
    log,
)
from splat.splat.telemetry_codec import Command

# Commands subject to repeat-detection. EXPERIMENT is excluded because back-to-back
# experiments with identical params are legitimate (e.g. repeated captures).
_DEDUP_COMMANDS = {"EXPERIMENT", "DATASET_PROCESSING", "DATASET_OD"}

# How long after completion to still reject an identical command (covers mainboard retransmit lag).
_DEDUP_GRACE_SECONDS = 60


def _command_key(command: Command):
    """Return a hashable fingerprint of a command's name + full argument set."""
    try:
        return (command.name, tuple(sorted(command.arguments.items())))
    except Exception:
        return None


class ExperimentThread(threading.Thread):
    """Consumes experiment requests and runs them asynchronously."""

    def __init__(self, stop_event: threading.Event):
        super().__init__(name="ExperimentThread", daemon=True)
        self.stop_event = stop_event

        self.experiment_runner = ExperimentRunner(stop_event=stop_event)
        # self.dataset_collection_runner = DatasetCollectionRunner(stop_event=stop_event)
        self.dataset_processing_runner = DatasetProcessingRunner(stop_event=stop_event)
        self.orbit_determination_runner = DatasetODRunner(stop_event=stop_event)

        self._experiment_handlers = {
            "EXPERIMENT": self.experiment_runner.run_experiment,
            # "DATASET_COLLECTION": self.dataset_collection_runner.run,
            "DATASET_PROCESSING": self.dataset_processing_runner.run,
            "DATASET_OD": self.orbit_determination_runner.run,
        }

        self._active_key = None       # key of the command currently executing
        self._last_key = None         # key of the last completed command
        self._last_completed_at = 0.0 # monotonic time of last completion

    def _is_duplicate(self, command: Command) -> bool:
        if command.name not in _DEDUP_COMMANDS:
            return False
        key = _command_key(command)
        if key is None:
            return False
        if key == self._active_key:
            log.warning(
                "Duplicate %s command received while identical command is still running — ignoring",
                command.name,
            )
            return True
        if key == self._last_key:
            age = time.monotonic() - self._last_completed_at
            if age < _DEDUP_GRACE_SECONDS:
                log.warning(
                    "Duplicate %s command received %.1fs after identical command completed — ignoring",
                    command.name,
                    age,
                )
                return True
        return False

    def run(self):
        log.info("Experiment thread started")
        while not self.stop_event.is_set():
            try:
                experiment_command = experiment_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not isinstance(experiment_command, Command):
                log.warning("Received non-Command in experiment_queue: %s", experiment_command)
                continue

            handler = self._experiment_handlers.get(experiment_command.name)
            if handler is None:
                log.debug("Ignoring unsupported experiment command: %s", experiment_command.name)
                continue

            if self._is_duplicate(experiment_command):
                continue

            key = _command_key(experiment_command)
            self._active_key = key
            try:
                handler(dict(experiment_command.arguments))
            finally:
                self._active_key = None
                if key is not None and experiment_command.name in _DEDUP_COMMANDS:
                    self._last_key = key
                    self._last_completed_at = time.monotonic()

            # now experiment is finished, do not need to do anything
            # command thread will receive the shutdown command, will ack it and will shut things down

