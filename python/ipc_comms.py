"""
Low-Level Communication layer - Named Pipe (FIFO)"

"""

from communication import PayloadComms


class PayloadCommsIPC(PayloadComms): # needed for local testing and SIL
    _connected = False
    _pckt_available = False

    @classmethod
    def connect(cls):
        cls._connected = True

    @classmethod
    def disconnect(cls):
        cls._connected = False

    @classmethod
    def send(cls, pckt):
        pass

    @classmethod
    def receive(cls):
        pass
