# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides helper functions for platforms.
"""

import queue
import threading

from serial import Serial
from serial.serialutil import SerialException


class UARTReader:
    """
    Reads bytes from the provided UART in a separate thread.
    """

    def __init__(self, *args, timeout: float = 0.5, **kwargs):
        """
        Constructs UART reader.

        Parameters
        ----------
        *args
            Arguments passed to Serial.
        timeout : float
            Data read timeout.
        **kwargs
            Keyword arguments passed to Serial.
        """
        self._queue = queue.Queue()

        self._conn = Serial(*args, **kwargs, timeout=timeout)
        self._stop = threading.Event()
        self._thread = None

        self.start()

    def __del__(self):
        self.stop()

    def _create_thread(self, conn: Serial):
        def _reader_thread():
            while not self._stop.is_set():
                try:
                    content = conn.read()
                except SerialException:
                    continue
                content += conn.read_all()
                if content:
                    self._queue.put(content)

        return threading.Thread(target=_reader_thread, daemon=True)

    def read(self, block: bool = False, timeout: float = None) -> bytes:
        """
        Returns data read from UART.

        Parameters
        ----------
        block : bool
            Whether the read should be blocking.
        timeout : float
            Read timeout.

        Returns
        -------
        bytes
            Data read from UART.
        """
        try:
            content = self._queue.get(block=block, timeout=timeout)
            return content
        except queue.Empty:
            return b""

    def start(self):
        """
        Starts reader.
        """
        if self._thread is None:
            self._stop.clear()
            self._thread = self._create_thread(self._conn)
            self._thread.start()

    def stop(self):
        """
        Stops reader.
        """
        if getattr(self, "_thread", None) is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
