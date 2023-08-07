# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Base class for bytes-based inference communication protocol.
"""

from abc import ABC
from time import time
from typing import Optional, Tuple
import selectors

from kenning.core.runtimeprotocol import RuntimeProtocol
from kenning.core.runtimeprotocol import ServerStatus
from kenning.core.runtimeprotocol import Message


class BytesBasedProtocol(RuntimeProtocol, ABC):

    arguments_structure = {
        'packet_size': {
            'description': 'The maximum size of the received packets, in bytes.',  # noqa: E50
            'type': int,
            'default': 4096
        },
        'endianness': {
            'description': 'The endianness of data to transfer',
            'default': 'little',
            'enum': ['big', 'little']
        }
    }

    def __init__(
            self,
            packet_size: int = 4096,
            endianness: str = 'little'):
        self.packet_size = packet_size
        self.endianness = endianness
        self.selector = selectors.DefaultSelector()
        self.input_buffer = b''
        super().__init__()

    def send_message(self, message: Message) -> bool:
        self.log.debug(f'Sending message {message}')
        return self.send_data(message.to_bytes())

    def receive_message(
            self,
            timeout: Optional[float] = None) -> Tuple[ServerStatus, Message]:
        server_status, data = self.gather_data(timeout)
        if data is None:
            return server_status, None

        self.input_buffer += data

        message, data_parsed = Message.from_bytes(
            self.input_buffer
        )
        if message is None:
            return ServerStatus.NOTHING, None

        self.input_buffer = self.input_buffer[data_parsed:]
        self.log.debug(f'Received message {message}')

        return ServerStatus.DATA_READY, message

    def gather_data(
            self,
            timeout: Optional[float] = None
            ) -> Tuple[ServerStatus, Optional[bytes]]:
        start_time = time()
        while True:
            events = self.selector.select(timeout=timeout)

            results = b''
            for key, mask in events:
                if mask & selectors.EVENT_READ:
                    callback = key.data
                    server_status, data = callback(key.fileobj, mask)
                    if (server_status == ServerStatus.CLIENT_DISCONNECTED
                            or data is None):
                        return server_status, None

                    results += data

            if results:
                return ServerStatus.DATA_READY, results
            elif not timeout or (time() - start_time > timeout):
                return ServerStatus.NOTHING, None

    def parse_message(self, message: bytes) -> Message:
        """
        Parses message from bytes.

        Parameters
        ----------
        message : bytes
            Received message.

        Returns
        -------
        Message :
            Parsed message.
        """
        return Message.from_bytes(message)[0]
