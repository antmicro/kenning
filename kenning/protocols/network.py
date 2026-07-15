# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
TCP-based inference communication protocol.
"""

import selectors
import socket
import threading
from typing import Callable, Dict, Optional

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
        "multi_client": {
            "description": "Allow multiple clients to connect simultaneously.",
            "type": bool,
            "default": False,
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
        multi_client: bool = False,
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
        multi_client : bool
            If True, server accepts multiple simultaneous client connections.
            Data is received from any connected client and sent to all clients.
        """
        self.host = host
        self.port = port
        self.collecteddata = bytes()
        self.serversocket = None
        self.socket = None
        self.multi_client = multi_client
        # multi client components
        self.sockets: Dict[str, socket.socket] = {}
        self.selector = selectors.DefaultSelector()
        self.acceptor_thread: Optional[threading.Thread] = None
        self.acceptor_running = False

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

        Raises
        ------
        ProtocolNotStartedError
            Server socket was not initialized.
        """
        if self.serversocket is None:
            raise ProtocolNotStartedError("Protocol not initialized.")
        self.serversocket.settimeout(timeout)
        try:
            if self.multi_client:
                self.serversocket.listen()
            else:
                self.serversocket.listen(1)
            sock, addr = self.serversocket.accept()
        except TimeoutError:
            return False

        if self.multi_client:
            sock.setblocking(False)
            addr_str = f"{addr[0]}:{addr[1]}"
            self.sockets[addr_str] = sock
            self.selector.register(sock, selectors.EVENT_READ, addr_str)
            KLogger.info(f"Connected client {addr}")
            sock.send(b"\x00")
            if self.client_connected_callback is not None:
                self.client_connected_callback()
            return True

        if self.socket is not None:
            KLogger.debug(f"Connection already established, rejecting {addr}")
            sock.close()
            return False
        self.socket = sock
        self.socket.settimeout(timeout)
        KLogger.info(f"Connected client {addr}")
        self.socket.send(b"\x00")
        if self.client_connected_callback is not None:
            self.client_connected_callback()
        return True

    def _acceptor_loop(self):
        """
        Thread function that is responsible for accepting new clients if
        multi_client is used.
        """
        while self.acceptor_running:
            self.serversocket.settimeout(1.0)
            try:
                sock, addr = self.serversocket.accept()
            except TimeoutError:
                continue
            except OSError:
                break

            sock.setblocking(False)
            addr_str = f"{addr[0]}:{addr[1]}"
            self.sockets[addr_str] = sock
            self.selector.register(sock, selectors.EVENT_READ, addr_str)
            KLogger.info(f"Connected client {addr}")
            sock.send(b"\x00")
            if self.client_connected_callback is not None:
                self.client_connected_callback()

    def initialize_server(
        self,
        client_connected_callback: Optional[Callable[[], None]] = None,
        client_disconnected_callback: Optional[Callable[[], None]] = None,
    ) -> bool:
        KLogger.debug(f"Initializing server at {self.host}:{self.port}")
        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.serversocket.settimeout(
            self.timeout if self.timeout > 0 else None
        )
        try:
            self.serversocket.bind((self.host, self.port))
            self.serversocket.listen(1)
        except OSError as execinfo:
            KLogger.error(f"{execinfo}", stack_info=True)
            self.serversocket.close()
            self.serversocket = None
            return False

        self.client_connected_callback = client_connected_callback
        self.client_disconnected_callback = client_disconnected_callback
        self.start()

        if self.multi_client:
            self.acceptor_running = True
            self.serversocket.listen()
            self.acceptor_thread = threading.Thread(
                target=self._acceptor_loop, daemon=True
            )
            self.acceptor_thread.start()

        return True

    def initialize_client(self) -> bool:
        KLogger.debug(f"Initializing client at {self.host}:{self.port}")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(self.timeout if self.timeout > 0 else None)
        self.socket.connect((self.host, self.port))

        try:
            data = self.socket.recv(1)
        except TimeoutError:
            KLogger.error("Server handshake waiting timeout.")
            data = b""
        except Exception as ex:
            data = b""
            KLogger.error(
                f"Unexpected exception during server handshake: {ex}"
            )

        if not data:
            self.socket.close()
            self.socket = None
            return False
        self.start()
        return True

    def _handle_client_disconnect(self, sock: socket.socket, addr_str: str):
        self.selector.unregister(sock)
        del self.sockets[addr_str]
        sock.close()
        if self.client_disconnected_callback is not None:
            self.client_disconnected_callback()
        KLogger.info(f"Client {addr_str} disconnected from the server.")

    def receive_data(self, timeout: Optional[float]) -> Optional[bytes]:
        if self.multi_client:
            if self.serversocket is None:
                raise ProtocolNotStartedError("Protocol not initialized.")

            events = self.selector.select(timeout)
            for key, mask in events:
                sock = key.fileobj
                addr_str = key.data
                try:
                    sock.settimeout(0)
                    data = sock.recv(self.packet_size)
                except TimeoutError:
                    continue
                except OSError:
                    data = b""

                if data == b"":
                    self._handle_client_disconnect(sock, addr_str)
                    continue

                if type(data) is bytes and len(data) > 0:
                    return data

            return None

        if self.socket is None:
            if self.serversocket is None:
                raise ProtocolNotStartedError("Protocol not initialized.")
            if not self.accept_client(timeout):
                return None

        if self.socket is None:
            KLogger.error(
                "Unable to receive data, connection socket is not available."
            )
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
        if self.multi_client:
            if not self.sockets:
                raise ProtocolNotStartedError("Protocol not initialized.")
            success = True
            for sock in self.sockets.values():
                remaining = data
                while remaining:
                    try:
                        ret = sock.send(remaining)
                    except (TimeoutError, OSError):
                        success = False
                        break
                    if ret <= 0:
                        success = False
                        break
                    remaining = remaining[ret:]
                if not success:
                    break
            return success

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

        if self.multi_client:
            self.acceptor_running = False
            if self.acceptor_thread and self.acceptor_thread.is_alive():
                self.acceptor_thread.join(timeout=2)

        if self.serversocket:
            self.serversocket.close()

        if self.multi_client:
            for sock in self.sockets.values():
                sock.close()
            self.sockets.clear()
            self.selector.close()
        elif self.socket:
            self.socket.close()

        self.socket = None
        self.serversocket = None
