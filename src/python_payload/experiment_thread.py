import queue
import threading
from experiment_runner import ExperimentRunner
from thread_shared import (
    experiment_queue,
    log,
)
from splat.splat.telemetry_codec import Command


class ExperimentThread(threading.Thread):
    """Consumes experiment requests and runs them asynchronously."""

    def __init__(self, stop_event: threading.Event):
        super().__init__(name="ExperimentThread", daemon=True)
        self.stop_event = stop_event
        self.experiment_runner = ExperimentRunner(stop_event=stop_event)
        self._experiment_handlers = {
            "EXPERIMENT": self.experiment_runner.run_experiment,
            "DATASET_COLLECTION": self.experiment_runner.run_dataset_collection,
            "DATASET_PROCESSING": self.experiment_runner.run_dataset_processing,
            "DATASET_OD": self.experiment_runner.run_dataset_od,
        }

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

            handler(dict(experiment_command.arguments))

