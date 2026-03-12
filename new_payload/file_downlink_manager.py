import os
import time
from dataclasses import dataclass

from splat.splat.telemetry_codec import Command, pack
from splat.splat.transport_layer import Transaction
from thread_shared import (
    create_transaction_ack_event,
    init_transaction_ack_event,
    log,
    tx_queue,
    update_last_batch_event,
    update_last_batch_lock,
    update_last_batch_shared,
)


@dataclass
class FileDownlinkResult:
    file_path: str
    success: bool
    reason: str = ""
    bursts_sent: int = 0


class FileDownlinkManager:
    """Robust sender-side file downlink manager."""

    def __init__(
        self,
        stop_event=None,
        burst_size: int = 32,
        ack_timeout_s: float = 10.0,
        ack_retry_interval_s: float = 3.0,
        confirm_timeout_s: float = 20.0,
        max_stalled_rounds: int = 5,
        completion_command_name: str = "RF_RESUME",
    ):
        self.stop_event = stop_event
        self.burst_size = max(1, int(burst_size))
        self.ack_timeout_s = max(0.1, float(ack_timeout_s))
        self.ack_retry_interval_s = max(0.1, float(ack_retry_interval_s))
        self.confirm_timeout_s = max(0.1, float(confirm_timeout_s))
        self.max_stalled_rounds = max(1, int(max_stalled_rounds))
        self.completion_command_name = completion_command_name

    def send_files(self, file_paths: list[str], stop_on_failure: bool = True) -> bool:
        """Send all files sequentially. Returns True only if all succeeded."""
        all_ok = True
        total = len(file_paths)

        for idx, file_path in enumerate(file_paths, start=1):
            log.info("Downlink file %s/%s: %s", idx, total, file_path)
            result = self.send_file(file_path)

            if result.success:
                log.info("Downlink completed for %s (%s bursts)", file_path, result.bursts_sent)
                continue

            all_ok = False
            log.error("Downlink failed for %s: %s", file_path, result.reason)
            if stop_on_failure:
                break

        if all_ok:
            self._send_completion_command()

        return all_ok

    def send_file(self, file_path: str, tid: int = 0) -> FileDownlinkResult:
        """Send a single file reliably, handling retries and batch confirmations."""
        if not os.path.exists(file_path):
            return FileDownlinkResult(file_path=file_path, success=False, reason="file_not_found")

        transaction = Transaction(tid, file_path, is_tx=True)
        self._clear_confirm_shared()

        # CREATE_TRANS phase
        create_command = Command("CREATE_TRANS")
        create_command.add_argument("tid", tid)
        create_command.add_argument("string_command", file_path)
        if not self._send_with_ack(create_command, create_transaction_ack_event, "CREATE_TRANS"):
            return FileDownlinkResult(file_path=file_path, success=False, reason="create_ack_timeout")

        # INIT_TRANS phase
        init_command = Command("INIT_TRANS")
        init_command.add_argument("tid", tid)
        init_command.add_argument("number_of_packets", transaction.number_of_packets)
        init_command.add_argument("hash_MSB", 0)
        init_command.add_argument("hash_middlesb", 0)
        init_command.add_argument("hash_LSB", 0)
        if not self._send_with_ack(init_command, init_transaction_ack_event, "INIT_TRANS"):
            return FileDownlinkResult(file_path=file_path, success=False, reason="init_ack_timeout")

        bursts_sent = 0
        stalled_rounds = 0
        previous_missing = len(transaction.missing_fragments)

        while len(transaction.missing_fragments) > 0:
            if self.stop_event is not None and self.stop_event.is_set():
                return FileDownlinkResult(file_path=file_path, success=False, reason="stop_requested", bursts_sent=bursts_sent)

            fragments = transaction.generate_x_packets(self.burst_size)
            if len(fragments) == 0:
                return FileDownlinkResult(file_path=file_path, success=False, reason="no_fragments_generated", bursts_sent=bursts_sent)

            for fragment in fragments:
                tx_queue.put(pack(fragment))
            bursts_sent += 1

            confirm = self._wait_confirm_last_batch(expected_tid=tid)
            if confirm is None:
                return FileDownlinkResult(file_path=file_path, success=False, reason="confirm_timeout", bursts_sent=bursts_sent)

            transaction.confirm_last_batch((confirm["MSB"], confirm["LSB"]))
            current_missing = len(transaction.missing_fragments)

            if current_missing == 0:
                break

            if current_missing >= previous_missing:
                stalled_rounds += 1
                log.warning(
                    "No missing-fragment progress for %s (missing=%s, stalled_round=%s/%s)",
                    file_path,
                    current_missing,
                    stalled_rounds,
                    self.max_stalled_rounds,
                )
                if stalled_rounds >= self.max_stalled_rounds:
                    return FileDownlinkResult(file_path=file_path, success=False, reason="stalled_progress", bursts_sent=bursts_sent)
            else:
                stalled_rounds = 0

            previous_missing = current_missing

        return FileDownlinkResult(file_path=file_path, success=True, bursts_sent=bursts_sent)

    def _send_with_ack(self, command: Command, ack_event, label: str) -> bool:
        """Send command and resend until ACK or timeout."""
        packet = pack(command)
        ack_event.clear()
        tx_queue.put(packet)

        deadline = time.monotonic() + self.ack_timeout_s
        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                return False

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False

            wait_s = min(self.ack_retry_interval_s, remaining)
            if ack_event.wait(timeout=wait_s):
                return True

            tx_queue.put(packet)
            log.warning("%s ACK not received yet, resending command", label)

    def _clear_confirm_shared(self):
        with update_last_batch_lock:
            update_last_batch_shared["data"] = None
        update_last_batch_event.clear()

    def _wait_confirm_last_batch(self, expected_tid: int):
        """Wait for CONFIRM_LAST_BATCH with matching tid."""
        deadline = time.monotonic() + self.confirm_timeout_s

        while True:
            if self.stop_event is not None and self.stop_event.is_set():
                return None

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None

            if not update_last_batch_event.wait(timeout=remaining):
                return None

            with update_last_batch_lock:
                data = update_last_batch_shared["data"]
                update_last_batch_shared["data"] = None
            update_last_batch_event.clear()

            if data is None:
                log.warning("Received CONFIRM_LAST_BATCH event with empty shared data")
                continue

            if int(data.get("tid", -1)) != expected_tid:
                log.warning(
                    "Ignoring CONFIRM_LAST_BATCH for tid=%s (expected tid=%s)",
                    data.get("tid"),
                    expected_tid,
                )
                continue

            return data

    def _send_completion_command(self):
        """Notify the satellite that the full multi-file downlink sequence has finished."""
        if not self.completion_command_name:
            return

        command = Command(self.completion_command_name)
        tx_queue.put(pack(command))
        log.info("Sent completion command to satellite: %s", self.completion_command_name)
