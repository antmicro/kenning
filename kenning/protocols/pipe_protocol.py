# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Named pipe-based inference communication protocol.
"""

import os
import select
import stat
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable, Optional, Union

from kenning.core.exceptions import ProtocolNotStartedError
from kenning.protocols.kenning_protocol import (
    KenningProtocol,
)
from kenning.utils.logger import KLogger


class PipeProtocol(KenningProtocol):
    """
    An Inter-Process-Communication based protocol, utilizing named pipe
    based communication.
    """

    arguments_structure = {
        "pipe_path": {
            "description": "Path to the pipe",
            "type": Union[Path, str],
            "required": True,
        },
    }

    def __init__(
        self,
        pipe_path: Union[Path, str],
        timeout: int = -1,
        error_recovery: bool = False,
        max_message_size: int = 1024,
    ):
        """
        Initializes PipeProtocol.

        Parameters
        ----------
        pipe_path: Union[Path, str]
            Path to the pipe.
        timeout : int
            Response receive timeout in seconds. If negative, then waits for
            responses forever.
        error_recovery: bool
            True if checksum verification and error recovery mechanisms are to
            be turned on.
        max_message_size : int
            Maximum size of a single protocol message in bytes.
        """
        if not isinstance(pipe_path, Path):
            pipe_path = Path(pipe_path)
        self.pipe_path = pipe_path.absolute()
        # fifo descriptor
        self.read_fd = None
        self.write_fd = None

        self._read_pipe_path = Path(f"{pipe_path}_server_read")
        self._write_pipe_path = Path(f"{pipe_path}_server_write")
        self._client_connected = False
        self._is_server = False
        self.wait_client_thread = None
        self._write_lock = Lock()
        self.client_connected_callback = None
        self.client_disconnected_callback = None

        super().__init__(timeout, error_recovery, max_message_size)

    def _open(self, path, options):
        fd = os.open(path, options)
        if fd == 0:
            return None
        return fd

    def is_pipe_open(self) -> bool:
        return self.read_fd is not None

    def connected(self) -> bool:
        if self._is_server:
            return self._client_connected
        return self.read_fd is not None and self.write_fd is not None

    def initialize_client(self) -> bool:
        KLogger.info("Initializing client side of pipeline communication")
        # open existing pipe
        KLogger.debug(f"Opening read pipe at {self._write_pipe_path}")
        self.read_fd = self._open(
            self._write_pipe_path, os.O_RDONLY | os.O_NONBLOCK
        )
        KLogger.debug(f"Opening write pipe at {self._read_pipe_path}")
        self.write_fd = self._open(self._read_pipe_path, os.O_WRONLY)

        if not self.connected():
            KLogger.warning("Failed to connect with server")
            return False

        self.start()

        return True

    def initialize_server(
        self,
        client_connected_callback: Optional[Callable[Any, None]] = None,
        client_disconnected_callback: Optional[Callable[None, None]] = None,
    ) -> bool:
        KLogger.info("Initializing server side of pipeline communication")
        self._is_server = True
        # create a fifo for writing
        KLogger.info(f"Creating a write fifo at path {self._write_pipe_path}")
        if not self._write_pipe_path.exists():
            os.mkfifo(
                self._write_pipe_path,
                stat.S_IROTH | stat.S_IWOTH | stat.S_IRUSR | stat.S_IWUSR,
            )
        else:
            return False
        # create a fifo for readings
        KLogger.info(f"Creating a read fifo at path {self._read_pipe_path}")
        if not self._read_pipe_path.exists():
            os.mkfifo(
                self._read_pipe_path,
                stat.S_IROTH | stat.S_IWOTH | stat.S_IRUSR | stat.S_IWUSR,
            )
        else:
            return False

        KLogger.debug(f"Opening read pipe at {self._read_pipe_path}")
        # open read pipe name
        self.read_fd = self._open(
            self._read_pipe_path, os.O_RDONLY | os.O_NONBLOCK
        )

        if self.read_fd == 0:
            return False

        self.client_connected_callback = client_connected_callback
        self.client_disconnected_callback = client_disconnected_callback
        self.start()
        self.attempt_to_connect(1)

        return True

    def _wait_for_client(self):
        with self._write_lock:
            try:
                self.write_fd = self._open(self._write_pipe_path, os.O_WRONLY)
                KLogger.info("Client connected.")
            except FileNotFoundError:
                return
            self.wait_client_thread = None
            self._client_connected = True
        if self.client_connected_callback is not None:
            self.client_connected_callback()

    def attempt_to_connect(self, timeout: Optional[float] = None) -> bool:
        # Initialize a thread that will wait until client is connected
        if self.wait_client_thread is None and not self._client_connected:
            KLogger.debug(f"Waiting for client at {self._write_pipe_path}")
            self.wait_client_thread = Thread(target=self._wait_for_client)
            self.wait_client_thread.daemon = True
            self.wait_client_thread.start()

        if timeout is None:
            timeout = -1

        res = self._write_lock.acquire(timeout=timeout)
        if res:
            self._write_lock.release()

        return res

    def _on_disconnect(self):
        if self._is_server:
            self._client_connected = False
            if self.client_disconnected_callback is not None:
                self.client_disconnected_callback()

    def send_data(self, data: bytes) -> bool:
        # wait for client to connect
        if self._is_server:
            self.attempt_to_connect(1)
        if not self.is_pipe_open():
            raise ProtocolNotStartedError("Pipe not open for write")
        if not self.connected():
            return False
        ret = False
        if self._write_lock.acquire(timeout=1):
            try:
                ret = os.write(self.write_fd, data) == len(data)
            except BrokenPipeError:
                self._write_lock.release()
                self._on_disconnect()
                raise ConnectionResetError()

            self._write_lock.release()

        return ret

    def receive_data(self, timeout: Optional[float] = None) -> Optional[bytes]:
        if self._is_server:
            self.attempt_to_connect(timeout)
        if not self.is_pipe_open():
            raise ProtocolNotStartedError("Pipe not open for read")
        if not self.connected():
            return None
        r, _, _ = select.select([self.read_fd], [], [], timeout)
        if self.read_fd in r:
            ret = os.read(self.read_fd, 1)
            if ret == b"":
                self._on_disconnect()
                return None
            else:
                return ret

    def _clean_pipes(self):
        if self.read_fd is not None:
            os.close(self.read_fd)
            self.read_fd = None
        if self.write_fd is not None:
            os.close(self.write_fd)
            self.write_fd = None
        if self._is_server:
            os.remove(self._read_pipe_path)
            os.remove(self._write_pipe_path)
            KLogger.debug("Pipes removed")

    def disconnect(self):
        self.stop()
        if not self.is_pipe_open():
            return
        if self._is_server:
            KLogger.debug("Disconnecting server...")
        else:
            KLogger.debug("Disconnecting...")
        self._clean_pipes()
        self._client_connected = False
