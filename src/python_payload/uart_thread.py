import queue
import threading
import time

import serial

from thread_shared import BAUD, PORT, MAX_PACKET_SIZE, TIMEOUT, log, rx_queue, tx_queue

IDLE_SLEEP_S = 0.001


class UartThread(threading.Thread):
    """Owns the serial port. Reads packets into rx_queue; drains tx_queue."""

    def __init__(self, stop_event: threading.Event):
        super().__init__(name="UartThread", daemon=True)
        self.stop_event = stop_event
        self._rx_buffer = bytearray()
        self._last_rx_time = None

    def run(self):
        log.info("Opening serial port %s @ %d baud", PORT, BAUD)
        try:
            with serial.Serial(PORT, BAUD, timeout=0) as ser:  # non-blocking
                while not self.stop_event.is_set():
                    did_work = self._read(ser)
                    did_work = self._write(ser) or did_work
                    if not did_work:
                        time.sleep(IDLE_SLEEP_S)
        except serial.SerialException as exc:
            log.error("Serial error: %s", exc)
            self.stop_event.set()

    def _read(self, ser: serial.Serial, max_packet_size=MAX_PACKET_SIZE):
        did_work = False
        available = ser.in_waiting
        if available > 0:
            chunk = ser.read(available)
            if chunk:
                self._rx_buffer.extend(chunk)
                self._last_rx_time = time.monotonic()
                did_work = True

        while len(self._rx_buffer) >= max_packet_size:
            data = bytes(self._rx_buffer[:max_packet_size])
            del self._rx_buffer[:max_packet_size]

            log.debug("RX %d bytes: %s", len(data), data.hex()[:20])
            rx_queue.put(data)
            did_work = True

            if len(self._rx_buffer) > 0:
                self._last_rx_time = time.monotonic()

        if self._rx_buffer and self._last_rx_time is not None:
            if time.monotonic() - self._last_rx_time > TIMEOUT:
                log.error(
                    "Dropping stale partial packet %d bytes: %s",
                    len(self._rx_buffer),
                    self._rx_buffer.hex()[:20],
                )
                self._rx_buffer.clear()
                self._last_rx_time = None
                did_work = True

        return did_work

    def _write(self, ser: serial.Serial):
        did_work = False
        while not tx_queue.empty():
            try:
                packet = tx_queue.get_nowait()
            except queue.Empty:
                break

            # Pad every outgoing packet to MAX_PACKET_SIZE bytes
            if len(packet) < MAX_PACKET_SIZE:
                packet = packet + b"\x00" * (MAX_PACKET_SIZE - len(packet))

            ser.write(packet)
            log.debug("TX %d bytes: %s", len(packet), packet.hex()[:20])
            did_work = True

        return did_work
