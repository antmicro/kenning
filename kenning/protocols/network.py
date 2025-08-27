# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
TCP-based inference communication protocol.
"""

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
        "packet_size": {
            "description": "Maximum number of bytes received at a time.",
            "type": int,
            "default": 4096,
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

    def accept_client(self, timeout: Optional[float]) -> bool:
        """
        Waits (blocking the current thread) on the new client and accepts or
        rejects the connection.

        Parameters
        ----------
        timeout: Optional[float]
            Maximum time waiting for the client in seconds. None means infinite
            timeout.

        Returns
        -------
        bool
            True if client connected successfully, False otherwise.
        """
        self.serversocket.settimeout(timeout)
        try:
            self.serversocket.listen(1)
            sock, addr = self.serversocket.accept()
        except TimeoutError:
            return False
        if self.socket is not None:
            KLogger.debug(f"Connection already established, rejecting {addr}")
            sock.close()
            return False
        self.socket = sock
        KLogger.info(f"Connected client {addr}")
        self.socket.send(b"\x00")
        if self.client_connected_callback is not None:
            self.client_connected_callback(addr)
        return True

    def initialize_server(
        self,
        # IP address will be passed to the 'client_connected_callback'
        client_connected_callback: Optional[Callable[Any, None]] = None,
        client_disconnected_callback: Optional[Callable[None, None]] = None,
    ) -> bool:
        KLogger.debug(f"Initializing server at {self.host}:{self.port}")
        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.serversocket.bind((self.host, self.port))
        except OSError as execinfo:
            KLogger.error(f"{execinfo}", stack_info=True)
            self.serversocket = None
            return False

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
        self.start()
        return True

    def receive_data(self, timeout: Optional[float]) -> Optional[bytes]:
        if self.socket is None:
            if self.serversocket is None:
                raise ProtocolNotStartedError("Protocol not initialized.")
            if not self.accept_client(timeout):
                return None
        self.socket.settimeout(timeout)
        data = None
        try:
            data = self.socket.recv(self.packet_size)
        except TimeoutError:
            return None
        if data == b"":
            self.socket.close()
            self.socket = None
            if self.serversocket is not None:
                if self.client_disconnected_callback is not None:
                    self.client_disconnected_callback()
                KLogger.info("Client disconnected from the server.")
            else:
                KLogger.error("Server abruptly disconnected.")
            return None
        elif type(data) is bytes:
            return data
        else:
            return None

    def send_data(self, data: bytes) -> bool:
        if self.socket is None:
            raise ProtocolNotStartedError("Protocol not initialized.")
        while True:
            ret = 0
            try:
                ret = self.socket.send(data)
            except TimeoutError:
                pass
            if ret < 0:
                return False
            data = data[ret:]
            if len(data) == 0:
                return True

    def disconnect(self):
        self.stop()
        if self.serversocket:
            self.serversocket.close()
        if self.socket:
            self.socket.close()
        self.socket = None
        self.serversocket = None
