"""

UART communication between mainboard and Jetson.

To be able to shutdown the jetson without running python with sudo
you need to make sure that the user is able to run sudo shutdown without needing to input the password
for that you need to run "sudo visudo" and add the following line to the end of the file

argus-payload ALL=(ALL) NOPASSWD: /usr/sbin/shutdown

if your user is not argus-payload, please change accordingly
"""

import threading
import time
import os

from command_thread import CommandThread
from experiment_thread import ExperimentThread
from telemetry_thread import TelemetryThread
from thread_shared import log
from uart_thread import UartThread


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
    
    if "remove_before_flight" in os.listdir("."):
        log.info("Running in development environment, skipping shutdown command")
        return
    
    # run shutdown command
    import subprocess
    subprocess.run(["sudo", "/usr/sbin/shutdown", "-h", "now"])

if __name__ == "__main__":
    main()
