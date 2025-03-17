# Low-Level Communication layer
import os
_PCKT_BUF_SIZE = 256

class PayloadComms:
    _connected = False
    _pckt_available = False
    _buffer = bytearray(_PCKT_BUF_SIZE)

    def __new__(cls, *args, **kwargs):
        if cls is PayloadComms:
            raise TypeError("PayloadComms is a static class and cannot be instantiated.")
        return super().__new__(cls)
    
    @classmethod
    def connect(cls):
        raise NotImplementedError("Subclass must implement connect()")

    @classmethod
    def disconnect(cls):
        raise NotImplementedError("Subclass must implement disconnect()")

    @classmethod
    def is_connected(cls):
        return cls._connected

    @classmethod
    def send(cls, pckt):
        raise NotImplementedError("Subclass must implement send()")

    @classmethod
    def receive(cls):
        raise NotImplementedError("Subclass must implement receive()")

    @classmethod
    def packet_available(cls):
        return cls._pckt_available
    