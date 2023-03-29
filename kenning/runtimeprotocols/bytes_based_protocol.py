# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Base class for bytes-based inference communication protocol.
"""

from typing import Optional, Tuple
import selectors

from kenning.core.runtimeprotocol import RuntimeProtocol
from kenning.core.runtimeprotocol import ServerStatus
from kenning.core.runtimeprotocol import Message
from kenning.core.runtimeprotocol import MSG_SIZE_LEN, MSG_TYPE_LEN
from kenning.utils.args_manager import add_parameterschema_argument


class BytesBasedProtocol(RuntimeProtocol):

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

    @classmethod
    def form_parameterschema(cls):
        """
        Creates schema for the RuntimeProtocol class.

        Returns
        -------
        Dict :
            schema for the class
        """
        parameterschema = cls._form_parameterschema()
        if cls.arguments_structure != BytesBasedProtocol.arguments_structure:
            add_parameterschema_argument(
                parameterschema,
                BytesBasedProtocol.arguments_structure
            )
        return parameterschema

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
        if len(self.input_buffer) < MSG_SIZE_LEN + MSG_TYPE_LEN:
            return ServerStatus.NOTHING, None

        data_to_load_len = int.from_bytes(
            self.input_buffer[:MSG_SIZE_LEN],
            byteorder=self.endianness,
            signed=False)
        if len(self.input_buffer) - MSG_SIZE_LEN < data_to_load_len:
            return ServerStatus.NOTHING, None

        message = Message.from_bytes(
            self.input_buffer[:MSG_SIZE_LEN + data_to_load_len]
        )
        self.input_buffer = self.input_buffer[MSG_SIZE_LEN + data_to_load_len:]
        self.log.debug(f'Received message {message}')

        return ServerStatus.DATA_READY, message

    def gather_data(
            self,
            timeout: Optional[float] = None
            ) -> Tuple[ServerStatus, Optional[bytes]]:
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

        if len(results) == 0:
            return ServerStatus.NOTHING, None

        return ServerStatus.DATA_READY, results

    def parse_message(self, message: bytes) -> Message:
        """
        Parses message from bytes.

        Parameters
        ----------
        message : bytes
            Received message

        Returns
        -------
        Message :
            Parsed message
        """
        return Message.from_bytes(message)
