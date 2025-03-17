# Low-Level Communication layer - UART

from communication import PayloadComms


class PayloadCommsUART(PayloadComms):
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
