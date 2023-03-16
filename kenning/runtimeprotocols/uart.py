# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
UART-based inference communication protocol.
"""

from typing import Any, Tuple, List, Optional, Dict
from pathlib import Path
import re
import json
import time
import serial
import numpy as np

from kenning.core.runtimeprotocol import RuntimeProtocol
from kenning.core.runtimeprotocol import MessageType
from kenning.core.runtimeprotocol import ServerStatus
from kenning.core.measurements import Measurements
from kenning.core.measurements import MeasurementsCollector
import kenning.utils.logger as logger


class UARTProtocol(RuntimeProtocol):

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
        },
        'endianness': {
            'description': 'The endianness of data to transfer',
            'default': 'little',
            'enum': ['big', 'little']
        }
    }

    def __init__(
            self,
            port: str,
            baudrate: int = 9600,
            endianness: str = 'little'):
        self.port = port
        self.baudrate = baudrate
        self.endianness = endianness
        self.collecteddata = bytes()
        self.log = logger.get_logger()
        super().__init__()

    @classmethod
    def from_argparse(cls, args):
        return cls(
            port=args.port,
            baudrate=args.baudrate,
            endianness=args.endianness
        )

    def initialize_client(self) -> bool:
        self.connection = serial.Serial(self.port, self.baudrate, timeout=10)
        return True

    def wait_for_activity(self) -> List[Tuple['ServerStatus', Any]]:
        self.log.info('wait_for_activity')

        try:
            return [ServerStatus.DATA_READY, self.receive_data()]
        except:
            return [ServerStatus.CLIENT_DISCONNECTED, None]

    def send_data(self, data: bytes) -> bool:
        length = (len(data)).to_bytes(4, self.endianness, signed=False)
        packet = length + data
        self.connection.write(packet)

    def receive_data(self) -> bytes:
        if not self.connection.is_open:
            return ServerStatus.CLIENT_DISCONNECTED, None
        length = int.from_bytes(
            self.connection.read(4),
            byteorder=self.endianness,
            signed=False
        )
        packet = self.connection.read(length)
        self.log.debug(f'PACKET: {packet}')
        return packet

    def send_message(self, messagetype: 'MessageType', data=bytes()) -> bool:
        """
        Sends message of a given type to the other side of connection.

        Parameters
        ----------
        messagetype : MessageType
            The type of the message
        data : bytes
            The additional data for a given message type

        Returns
        -------
        bool :
            True if succeded
        """
        mt = messagetype.to_bytes()
        return self.send_data(mt + data)

    def parse_message(self, message):
        mt = MessageType.from_bytes(message[:2], self.endianness)
        data = message[2:]
        return mt, data

    def receive_confirmation(self) -> Tuple[bool, Optional[bytes]]:
        """
        Waits until the OK message is received.

        Method waits for the OK message from the other side of connection.

        Returns
        -------
        bool :
            True if OK received, False otherwise
        """
        while True:
            status, data = self.wait_for_activity()
            if status == ServerStatus.DATA_READY:
                typ, dat = self.parse_message(data)
                self.log.debug(f'TYPE {typ}')
                if typ == MessageType.ERROR:
                    self.log.error('Error during uploading input')
                    return False, None
                if typ != MessageType.OK:
                    self.log.error('Unexpected message')
                    return False, None
                self.log.debug('Upload finished successfully')
                return True, dat
            elif status == ServerStatus.CLIENT_DISCONNECTED:
                self.log.error('Client is disconnected')
                return False, None
            elif status == ServerStatus.DATA_INVALID:
                self.log.error('Received invalid packet')
                return False, None

    def upload_input(self, data: bytes) -> bool:
        self.log.debug('Uploading input')
        self.send_message(MessageType.DATA, data)
        return self.receive_confirmation()[0]

    def upload_model(self, path: Path) -> bool:
        self.log.debug('Uploading model')
        with open(path, 'rb') as model_file:
            data = model_file.read()
            self.send_message(MessageType.MODEL, data)
            return self.receive_confirmation()[0]

    def request_processing(self) -> bool:
        self.log.debug('Requesting processing')
        self.send_message(MessageType.PROCESS)
        ret = self.receive_confirmation()[0]
        if not ret:
            return False
        start = time.perf_counter()
        ret = self.receive_confirmation()[0]
        if not ret:
            return False
        duration = time.perf_counter() - start
        measurementname = 'protocol_inference_step'
        MeasurementsCollector.measurements += {
            measurementname: [duration],
            f'{measurementname}_timestamp': [time.perf_counter()]
        }
        return True

    def download_output(self) -> Tuple[bool, Optional[bytes]]:
        self.log.debug('Downloading output')
        self.send_message(MessageType.OUTPUT)
        return self.receive_confirmation()

    def download_statistics(self) -> 'Measurements':
        raise NotImplementedError

    def disconnect(self):
        self.connection.close()
