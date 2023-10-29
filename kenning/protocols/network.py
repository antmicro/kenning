# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
TCP-based inference communication protocol.
"""

import selectors
import socket
from typing import Optional, Tuple

from kenning.core.protocol import ServerStatus
from kenning.protocols.bytes_based_protocol import BytesBasedProtocol
from kenning.utils.logger import KLogger


class NetworkProtocol(BytesBasedProtocol):
    """
    A TCP-based protocol.

    Protocol is implemented using BSD sockets and selectors-based pooling.
    """

    arguments_structure = {
        "host": {
            "description": "The address to the target device",
            "type": str,
            "required": True,
        },
        "port": {
            "description": "The port for the target device",
            "type": int,
            "required": True,
        },
    }

    def __init__(
        self,
        host: str,
        port: int,
        packet_size: int = 4096,
        endianness: str = "little",
    ):
        """
        Initializes NetworkProtocol.

        Parameters
        ----------
        host : str
            Host for the TCP connection.
        port : int
            Port for the TCP connection.
        packet_size : int
            Receive packet sizes.
        endianness : str
            Endianness of the communication.
        """
        self.host = host
        self.port = port
        self.collecteddata = bytes()
        self.serversocket = None
        self.socket = None
        super().__init__(packet_size=packet_size, endianness=endianness)

    def accept_client(
        self, socket: socket.socket, mask: int
    ) -> Tuple["ServerStatus", Optional[bytes]]:
        """
        Accepts the new client.

        Parameters
        ----------
        socket : socket.socket
            New client's socket.
        mask : int
            Selector mask. Not used.

        Returns
        -------
        Tuple['ServerStatus', Optional[bytes]]
            Client accepted status and None.
        """
        sock, addr = socket.accept()
        if self.socket is not None:
            KLogger.debug(f"Connection already established, rejecting {addr}")
            sock.close()
            return ServerStatus.CLIENT_IGNORED, None
        else:
            self.socket = sock
            KLogger.info(f"Connected client {addr}")
            self.socket.setblocking(False)
            self.selector.register(
                self.socket,
                selectors.EVENT_READ | selectors.EVENT_WRITE,
                self.receive_data,
            )
            return ServerStatus.CLIENT_CONNECTED, None

    def initialize_server(self) -> bool:
        KLogger.debug(f"Initializing server at {self.host}:{self.port}")
        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.serversocket.setblocking(False)
        try:
            self.serversocket.bind((self.host, self.port))
        except OSError as execinfo:
            KLogger.error(f"{execinfo}", stack_info=True)
            self.serversocket = None
            return False
        self.serversocket.listen(1)
        self.selector.register(
            self.serversocket, selectors.EVENT_READ, self.accept_client
        )
        return True

    def initialize_client(self) -> bool:
        KLogger.debug(f"Initializing client at {self.host}:{self.port}")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        self.selector.register(
            self.socket,
            selectors.EVENT_READ | selectors.EVENT_WRITE,
            self.receive_data,
        )
        return True

    def receive_data(
        self, socket: socket.socket, mask: int
    ) -> Tuple[ServerStatus, Optional[bytes]]:
        data = self.socket.recv(self.packet_size)
        if not data:
            KLogger.info("Client disconnected from the server")
            self.selector.unregister(self.socket)
            self.socket.close()
            self.socket = None
            return ServerStatus.CLIENT_DISCONNECTED, None
        else:
            return ServerStatus.DATA_READY, data

    def wait_send(self, data: bytes) -> int:
        """
        Wrapper for sending method that waits until write buffer is ready for
        new data.

        Parameters
        ----------
        data : bytes
            Data to send.

        Returns
        -------
        int
            The number of bytes sent.
        """
        if self.socket is None:
            return -1

        ret = self.socket.send(data)
        while True:
            events = self.selector.select(timeout=1)
            for key, mask in events:
                if key.fileobj == self.socket and mask & selectors.EVENT_WRITE:
                    return ret

    def send_data(self, data: bytes):
        index = 0
        while index < len(data):
            ret = self.wait_send(data[index:])
            if ret < 0:
                return False
            index += ret
        return True

    def disconnect(self):
        if self.serversocket:
            self.serversocket.close()
        if self.socket:
            self.socket.close()
