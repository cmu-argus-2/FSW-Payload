import queue
import threading
import json

from splat.splat.telemetry_codec import Ack, Command, Report, pack, unpack
from thread_shared import experiment_queue, log, rx_queue, tx_queue
from thread_shared import create_transaction_ack_event, init_transaction_ack_event, update_last_batch_event
from thread_shared import update_last_batch_lock, update_last_batch_shared



class CommandThread(threading.Thread):
    """Consumes raw packets from rx_queue, decodes them, and dispatches."""

    def __init__(self, stop_event: threading.Event, telemetry_thread=None, experiment_thread=None):
        super().__init__(name="CommandThread", daemon=True)
        self.stop_event = stop_event
        self.telemetry_thread = telemetry_thread
        self.experiment_thread = experiment_thread



    def run(self):
        log.info("Command thread started")
        while not self.stop_event.is_set():
            try:
                data = rx_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._handle_packet(data)

    def _handle_packet(self, data: bytes):
        try:
            callsign, message = unpack(data)
        except Exception as exc:
            log.warning("Failed to unpack packet: %s", exc)
            return

        log.info("Received message from '%s': %s", callsign, message)

        if isinstance(message, Command):
            log.info("Handling command: %s", message.name)
            self._dispatch_command(message)
            return
        
        if isinstance(message, Ack):
            log.info("Handling ACK: %s", message)
            self.dispatch_ack(message)
            return

        # we receive something that was not a command or a ack
        log.warning("Unexpected message type: %s", type(message))
        

    def dispatch_ack(self, ack: Ack):
        """
        This is the function that will be used to deal with ack
        it will simply set the ack variable for that given function true
        """
        if ack.cmd_id == 15:
            # it was a ping command
            if create_transaction_ack_event.is_set():
                log.error("CREATE_TRANS ACK OVERRIDDEN")
            create_transaction_ack_event.set()
            log.info("create_transaction_ack set to true")
            return
            
        
        if ack.cmd_id == 16:
            # it was a ping command
            if init_transaction_ack_event.is_set():
                log.error("INIT_TRANS ACK OVERRIDDEN")
            init_transaction_ack_event.set()
            log.info("init_transaction_ack set to true")
            return

    def _dispatch_command(self, command: Command):
        handler = self._handlers.get(command.name)
        if handler:
            handler(self, command)
        else:
            log.warning("No handler for command: %s", command.name)

    def _handle_ping(self, command: Command):
        log.info("PING  ts=%s", command.arguments.get("ts"))
        self._send_ack(command)

    def _handle_experiment(self, command: Command):
        log.info("EXPERIMENT  args=%s", command.arguments)
        experiment_queue.put(dict(command.arguments))
        self._send_ack(command)
    
    def _handle_request_tm_payload(self, command: Command):
        if self.telemetry_thread is None:
            log.warning("Telemetry thread not available")
            self._send_ack(command, ack_args={"error": "telemetry thread unavailable"})
            return

        telemetry_data = self.telemetry_thread.get_latest_data()
        if not telemetry_data:
            telemetry_data = {"status": "no telemetry collected yet"}

        log.info("REQUEST_TM_PAYLOAD: %s", telemetry_data)

        
        # create the payload telemetry report
        report = Report("TM_PAYLOAD")
        
        # add the variables
        for var_name, value in telemetry_data.items():
            report.add_variable(var_name, "PAYLOAD_TM", value)
            
        # packet the data
        packet = pack(report)
        # send the data
        tx_queue.put(packet)

        self._send_ack(command, ack_args=True)
    
    # def _handle_create_trans(self, command: Command):
    #     """
    #     Received a command to create a transaction
    #     TODO - SHOULD DELETE THIS, THIS IS OLD STUFF
    #     """
    #     file_path = command.arguments["string_command"].rstrip("\x00")  # remove tranling 0s
    #     tid = command.arguments["tid"]
    #     # 1. check if the file exists and get the path to the file
    #     # 2. create a transaction in the transaction manager
    #     transaction = self.TM.create_transaction(file_path=file_path, tid=tid, is_tx=True)
    #     if transaction is None:
    #         log.error(f"Unable to create transaction {tid}")
    #         return ["transaction_creation_failed"]

    #     # 3. generate init transaction packet
    #     cmd = Command("INIT_TRANS")
    #     tid = transaction.tid
    #     n_packets = transaction.number_of_packets
    #     hash_MSB, hash_msb, hash_LSB = transaction.get_hash_as_integers()

    #     cmd.set_arguments(tid, n_packets, hash_MSB, hash_msb, hash_LSB)
      
    #     # 4. send the init trans command
    #     tx_queue.put(pack(cmd))

    def _handle_update_last_batch(self, command: Command):
        """
        This command will be sent after the satellite has received all the packets in listen mode
        it will update the local missing fragments list, so the jetson will send new packets in the
        next burst
        """
        handled = False

        try:
            data = {
                "tid": int(command.arguments.get("tid")),
                "MSB": int(command.arguments.get("MSB")),
                "LSB": int(command.arguments.get("LSB")),
            }
        except (TypeError, ValueError) as exc:
            log.error("Invalid UPDATE_LAST_BATCH arguments: %s", exc)
            self._send_ack(command, ack_args={"handled": handled, "error": "invalid_arguments"})
            return

        with update_last_batch_lock:
            if update_last_batch_event.is_set():
                update_last_batch_shared["overrun_count"] += 1
                log.warning(
                    "UPDATE_LAST_BATCH overrun: previous update not consumed yet (count=%s)",
                    update_last_batch_shared["overrun_count"],
                )

            update_last_batch_shared["data"] = data

        update_last_batch_event.set()
        handled = True
        self._send_ack(command, ack_args={"handled": handled})
        

    def _send_ack(self, command: Command, ack_args=None):
        if isinstance(ack_args, dict):
            ack_args = json.dumps(ack_args, separators=(",", ":"))

        ack = Ack(0, command.command_id, ack_args=ack_args)
        packet = pack(ack)
        tx_queue.put(packet)
        log.debug("Queued ACK for command_id=%s", command.command_id)
    
    
    _handlers = {
        "PING": _handle_ping,
        "EXPERIMENT": _handle_experiment,
        "REQUEST_TM_PAYLOAD": _handle_request_tm_payload,
        "CONFIRM_LAST_BATCH": _handle_update_last_batch,
        # "CREATE_TRANS": _handle_create_trans
    }
