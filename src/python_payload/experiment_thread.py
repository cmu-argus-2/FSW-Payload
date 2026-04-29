import queue
import threading
from experiment_runner import ExperimentRunner
from orbit_determination_runner import DatasetCollectionRunner
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
        self.dataset_collection_runner = DatasetCollectionRunner(stop_event=stop_event)
        
        self._experiment_handlers = {
            "EXPERIMENT": self.experiment_runner.run_experiment,
            "DATASET_COLLECTION": self.dataset_collection_runner.run,
            "DATASET_PROCESSING": self.dataset_collection_runner.run,
            "DATASET_OD": self.dataset_collection_runner.run,
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
            
            # experiment has finished
            # if it failed before download it will not quit the program
            self.stop_event.set()

