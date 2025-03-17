"""
Low-Level Communication layer - Named Pipe (FIFO)"

This is the low-level communication layer for the Payload process using named pipes (FIFO). 
This is used in lieu of of UART for Software-in-the-Loop (SIL) testing and local test 
scripts (on the host device) with the Payload.

Author: Ibrahima Sory Sow
"""

import os
import select

from communication import PayloadCommunicationInterface

# Named pipe paths (make sure this corresponds to the CMAKE compile definitions)
FIFO_IN = "/tmp/payload_fifo_in"  # Payload reads from this, external process writes to it
FIFO_OUT = "/tmp/payload_fifo_out"  # Payload writes to this, external process reads from it


class PayloadIPC(PayloadCommunicationInterface):  # needed for local testing and SIL
    _connected = False
    _pckt_available = False
    _pipe_in = None  # File descriptor for writing
    _pipe_out = None  # File descriptor for reading

    @classmethod
    def connect(cls):
        """Establishes a connection using named pipes (FIFO)."""
        if cls._connected:
            return

        # Ensure FIFOs exist
        for fifo in [FIFO_IN, FIFO_OUT]:
            if not os.path.exists(fifo):
                try:
                    os.mkfifo(fifo, 0o666)
                except OSError as e:
                    print(f"Error creating FIFO {fifo}: {e}")
                    return

        # Open FIFOs
        try:
            cls._pipe_in = open(FIFO_IN, "w", buffering=1)  # Line-buffered write
            cls._pipe_out = os.open(FIFO_OUT, os.O_RDONLY | os.O_NONBLOCK)  # Non-blocking read
            cls._connected = True
            print("[INFO] PayloadIPC connected.")
        except OSError as e:
            print(f"[ERROR] Failed to open FIFOs: {e}")
            cls.disconnect()

    @classmethod
    def disconnect(cls):
        """Closes the named pipe connections."""
        if not cls._connected:
            return

        if cls._pipe_in:
            cls._pipe_in.close()
            cls._pipe_in = None

        if cls._pipe_out:
            os.close(cls._pipe_out)
            cls._pipe_out = None

        cls._connected = False
        print("[INFO] PayloadIPC disconnected.")

    @classmethod
    def send(cls, pckt: bytes):
        """Sends a packet (bytes) via the named pipe."""
        if not cls._connected or cls._pipe_in is None:
            print("[ERROR] Attempt to send while not connected.")
            return False

        try:
            cls._pipe_in.write(pckt.decode() + "\n")  # convert bytes to string
            cls._pipe_in.flush()
            return True
        except OSError as e:
            print(f"[ERROR] Failed to write to FIFO: {e}")
            return False

    @classmethod
    def receive(cls) -> bytes:
        """Receives a packet from the named pipe (FIFO_OUT)."""
        if not cls._connected or cls._pipe_out is None:
            print("[ERROR] Attempt to receive while not connected.")
            return b""

        try:
            rlist, _, _ = select.select([cls._pipe_out], [], [], 0.5)  # 100ms timeout
            if cls._pipe_out in rlist:
                data = os.read(cls._pipe_out, 512).strip()
                if data:
                    return data
        except OSError as e:
            print(f"[ERROR] Failed to read from FIFO: {e}")

        return b""  # No data available


if __name__ == "__main__":

    # Test script for Payload comms

    PayloadIPC.connect()

    PayloadIPC.send(b"3")  # Request telemetry data

    response = PayloadIPC.receive()
    if response:
        print(f"Received: {response.decode()}")

    PayloadIPC.disconnect()
