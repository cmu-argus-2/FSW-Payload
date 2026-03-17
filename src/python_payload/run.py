"""UART communication between mainboard and Jetson."""

import threading
import time

from command_thread import CommandThread
from experiment_thread import ExperimentThread
from telemetry_thread import TelemetryThread
from thread_shared import log, PayloadState, state_manager
from uart_thread import UartThread

import camera_driver


def main():
    stop_event = threading.Event()   # used to stop all the threads

    experiment_thread = ExperimentThread(stop_event)
    telemetry_thread = TelemetryThread(stop_event)
    uart_thread = UartThread(stop_event)
    command_thread = CommandThread(
        stop_event,
        telemetry_thread=telemetry_thread,
        experiment_thread=experiment_thread,
    )

    uart_thread.start()         # read and send data to uart
    command_thread.start()      # interpret the data received from uart and convert them into commands
    experiment_thread.start()   # perform the experiment when the command has been received
    telemetry_thread.start()    # gather telemetry data while running the program

    log.info("Running — press Ctrl+C to stop")
    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        log.info("Shutdown requested")
        stop_event.set()

    experiment_thread.join(timeout=2)
    telemetry_thread.join(timeout=2)
    uart_thread.join(timeout=2)
    command_thread.join(timeout=2)
    log.info("Shutdown complete")


main()
if __name__ == "__main__":
    main()
