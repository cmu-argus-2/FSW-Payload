"""
Payload Control Interface

This module defines the Payload Controller class, which is responsible for managing the main interface between
the host and the Payload. 

Author: Ibrahima Sory Sow

"""

from protocol import Encoder, Decoder
from definitions import CommandID, ErrorCodes

_PING_RESP_VALUE = 0x60
class PayloadTM:  # Simple data structure holder
    # System part
    SYSTEM_TIME: int = 0
    SYSTEM_UPTIME: int = 0
    LAST_EXECUTED_CMD_TIME: int = 0
    LAST_EXECUTED_CMD_ID: int = 0
    PAYLOAD_STATE: int = 0
    ACTIVE_CAMERAS: int = 0
    CAPTURE_MODE: int = 0
    CAM_STATUS: list = [0] * 4
    TASKS_IN_EXECUTION: int = 0
    DISK_USAGE: int = 0
    LATEST_ERROR: int = 0
    # Tegrastats part
    TEGRASTATS_PROCESS_STATUS: bool = False
    RAM_USAGE: int = 0
    SWAP_USAGE: int = 0
    ACTIVE_CORES: int = 0
    CPU_LOAD: list = [0] * 6
    GPU_FREQ: int = 0
    CPU_TEMP: int = 0
    GPU_TEMP: int = 0
    VDD_IN: int = 0
    VDD_CPU_GPU_CV: int = 0
    VDD_SOC: int = 0

class PayloadController:

    communication_interface = None

    @classmethod
    def initialize(cls, communication_interface):
        cls.communication_interface = communication_interface
        cls.communication_interface.connect()

    @classmethod
    def deinitialize(cls): 
        cls.communication_interface.disconnect()

    @classmethod 
    def ping(cls):
        cls.communication_interface.send(Encoder.encode_ping())

        resp = cls.communication_interface.receive()
        if resp:
            return Decoder.decode(resp) == _PING_RESP_VALUE
        return False

    @classmethod
    def shutdown(cls):
        pass

    @classmethod
    def request_telemetry(cls):
        pass