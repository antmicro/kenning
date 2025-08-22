# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
TCP-based inference communication protocol.
"""

import selectors
import socket
from typing import Any, Callable, Optional

from kenning.protocols.kenning_protocol import (
    KenningProtocol,
    ProtocolNotStartedError,
)
from kenning.utils.logger import KLogger


class NetworkProtocol(KenningProtocol):
    """
    A TCP-based protocol.

    Protocol is implemented using BSD sockets and selectors-based pooling.
    """

    arguments_structure = {
        "host": {
            "description": "The address to the target device",
            "type": str,
            "default": "localhost",
        },
        "port": {
            "description": "The port for the target device",
            "type": int,
            "default": 12345,
        },
    }

    def __init__(
        self,
        host: str = "localhost",
        port: int = 12345,
        timeout: int = -1,
        error_recovery: bool = False,
        packet_size: int = 4096,
        max_message_size: int = 1024 * 1024,
    ):
        """
        Initializes NetworkProtocol.

        Parameters
        ----------
        host : str
            Host for the TCP connection.
        port : int
            Port for the TCP connection.
        timeout : int
            Response receive timeout in seconds. If negative, then waits for
            responses forever.
        error_recovery: bool
            True if checksum verification and error recovery mechanisms are to
            be turned on.
        packet_size : int
            Receive packet sizes.
        max_message_size : int
            Maximum size of a single protocol message in bytes.
        """
        self.host = host
        self.port = port
        self.collecteddata = bytes()
        self.serversocket = None
        self.socket = None
        self.packet_size = packet_size
        self.client_connected_callback = None
        self.client_disconnected_callback = None
        super().__init__(timeout, error_recovery, max_message_size)

    def accept_client(self, socket: socket.socket, mask: int) -> bool:
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
        bool
            True if client connected successfully, False otherwise.
        """
        sock, addr = socket.accept()
        if self.socket is not None:
            KLogger.debug(f"Connection already established, rejecting {addr}")
            sock.close()
        else:
            self.socket = sock
            KLogger.info(f"Connected client {addr}")
            self.socket.setblocking(True)
            self.socket.send(b"\x00")
            self.selector.register(
                self.socket,
                selectors.EVENT_READ | selectors.EVENT_WRITE,
                self.receive_data,
            )
            if self.client_connected_callback is not None:
                self.client_connected_callback(addr)
        return None

    def initialize_server(
        self,
        # IP address will be passed to the 'client_connected_callback'
        client_connected_callback: Optional[Callable[Any, None]] = None,
        client_disconnected_callback: Optional[Callable[None, None]] = None,
    ) -> bool:
        KLogger.debug(f"Initializing server at {self.host}:{self.port}")
        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.serversocket.setblocking(True)
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
        self.client_connected_callback = client_connected_callback
        self.client_disconnected_callback = client_disconnected_callback
        self.start()
        return True

    def initialize_client(self) -> bool:
        KLogger.debug(f"Initializing client at {self.host}:{self.port}")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        if not self.socket.recv(1):
            self.socket = None
            return False
        self.selector.register(
            self.socket,
            selectors.EVENT_READ | selectors.EVENT_WRITE,
            self.receive_data,
        )
        self.start()
        return True

    def receive_data(
        self, socket: socket.socket, mask: int
    ) -> Optional[bytes]:
        if self.socket is None:
            raise ProtocolNotStartedError("Protocol not initialized.")
        data = self.socket.recv(self.packet_size)
        if not data:
            KLogger.info("Client disconnected from the server")
            self.selector.unregister(self.socket)
            self.socket.close()
            self.socket = None
            if self.client_disconnected_callback is not None:
                self.client_disconnected_callback()
            return None
        else:
            return data

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

        Raises
        ------
        ProtocolNotStartedError
            The protocol has not been initialized, or it has been initialized
            as server, but no client has connected.
        """
        if self.socket is None:
            raise ProtocolNotStartedError("Protocol not initialized.")
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
        self.stop()
        if self.serversocket:
            self.serversocket.close()
        if self.socket:
            self.socket.close()
        self.socket = None
        self.serversocket = None
