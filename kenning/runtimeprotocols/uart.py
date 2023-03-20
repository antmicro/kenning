# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
UART-based inference communication protocol.
"""

from typing import Tuple, Optional
import selectors
import serial

from kenning.core.runtimeprotocol import RuntimeProtocol
from kenning.core.runtimeprotocol import ServerStatus
import kenning.utils.logger as logger


class UARTProtocol(RuntimeProtocol):
    """
    An UART-base runtime protocol.

    It is implemented using pyserial.
    """

    arguments_structure = {
        'port': {
            'description': 'The target device name',
            'type': str,
            'required': True
        },
        'baudrate': {
            'description': 'The baud rate',
            'type': int,
            'default': 9600
        }
    }

    def __init__(
            self,
            port: str,
            baudrate: int = 9600,
            packet_size: int = 4096,
            endianness: str = 'little'):
        """
        Initializes UARTProtocol.

        Parameters
        ----------
        port : str
            UART port
        baudrate : int
            UART baudrate
        endiannes : str
            endianness of the communication
        """
        self.port = port
        self.baudrate = baudrate
        self.collecteddata = bytes()
        self.connection = None
        self.log = logger.get_logger()
        super().__init__(packet_size=packet_size, endianness=endianness)

    @classmethod
    def from_argparse(cls, args):
        return cls(
            port=args.port,
            baudrate=args.baudrate,
            endianness=args.endianness
        )

    def initialize_client(self) -> bool:
        self.connection = serial.Serial(self.port, self.baudrate, timeout=1)
        self.selector.register(
            self.connection,
            selectors.EVENT_READ | selectors.EVENT_WRITE,
            self.receive_data
        )
        return self.connection.is_open

    def send_data(self, data: bytes) -> bool:
        if self.connection is None or not self.connection.is_open:
            return False
        self.connection.write(data)

    def receive_data(
            self,
            connection: serial.Serial,
            mask: int) -> Tuple[ServerStatus, Optional[bytes]]:
        if self.connection is None or not self.connection.is_open:
            return ServerStatus.CLIENT_DISCONNECTED, None

        data = self.connection.read(self.packet_size)

        return ServerStatus.DATA_READY, data

    def disconnect(self):
        if self.connection is not None or self.connection.is_open:
            self.connection.close()
