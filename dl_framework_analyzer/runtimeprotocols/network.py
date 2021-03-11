import socket

from dl_framework_analyzer.core.runtimeprotocol import RuntimeProtocol


class NetworkProtocol(object):
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )

    @classmethod
    def form_argparse(cls):
        parser, group = super().form_argparse()
        group.add_argument(
            '--host',
            help='The address to the target device',
            type=str
        )
        group.add_argument(
            '--port',
            help='The port for the target device',
            type=int
        )
        return parser, group

    @classmethod
    def from_argparse(cls, args):
        return cls(host, port)

    def connect(self):
        pass
