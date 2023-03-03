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
import selectors
import numpy as np

from kenning.core.runtimeprotocol import RuntimeProtocol
from kenning.core.runtimeprotocol import MessageType
from kenning.core.runtimeprotocol import ServerStatus
from kenning.core.measurements import Measurements
from kenning.core.measurements import MeasurementsCollector
import kenning.utils.logger as logger


MAX_MODEL_INPUT_NUM = 2
MAX_MODEL_INPUT_DIM = 4
MAX_MODEL_OUTPUTS = 12
MAX_LENGTH_ENTRY_FUNC_NAME = 20
MAX_LENGTH_MODEL_NAME = 20


def io_spec_to_iree_model_struct(
        io_spec: Dict[str, Any],
        entry_func: str,
        model_name: str,
        byteorder: str = 'little') -> bytes:
    input_shape = [inp['shape'] for inp in io_spec['inputs']]
    output_length = [
        int(np.prod(outp['shape'])) for outp in io_spec['outputs']
    ]
    dtype = io_spec['inputs'][0]['dtype']

    dtype_size = re.findall(r'\d+', dtype)
    assert len(dtype_size) == 1, f'Wrong dtype {dtype}'
    dtype_size_bytes = int(dtype_size[0])//8
    dtype = dtype[0] + dtype_size[0]

    num_input = len(input_shape)
    num_input_dim = [len(inp) for inp in input_shape]
    input_length = [int(np.prod(inp)) for inp in input_shape]
    input_size_bytes = [dtype_size_bytes for _ in input_shape]
    num_output = len(output_length)
    output_size_bytes = dtype_size_bytes

    def int_to_bytes(num: int) -> bytes:
        return num.to_bytes(4, byteorder=byteorder, signed=False)

    def array_to_bytes(a: List, shape: Tuple) -> np.ndarray:
        a_np = np.array(a, dtype=np.uint32)
        print(shape)
        print([(0, shape[i] - a_np.shape[i]) for i in range(a_np.ndim)])
        a_pad = np.pad(
            a_np,
            [(0, shape[i] - a_np.shape[i]) for i in range(a_np.ndim)]
        )
        return a_pad.astype(np.uint32).tobytes('C')

    def str_to_bytes(s: str, length: int) -> bytes:
        return s.ljust(length)[:length].encode('ascii')

    # input spec
    result = bytes()
    result += int_to_bytes(num_input)
    result += array_to_bytes(num_input_dim, (MAX_MODEL_INPUT_NUM,))
    result += array_to_bytes(
        input_shape,
        (MAX_MODEL_INPUT_NUM, MAX_MODEL_INPUT_DIM)
    )
    result += array_to_bytes(input_length, (MAX_MODEL_INPUT_NUM,))
    result += array_to_bytes(input_size_bytes, (MAX_MODEL_INPUT_NUM,))

    # output spec
    result += int_to_bytes(num_output)
    result += array_to_bytes(output_length, (MAX_MODEL_OUTPUTS,))
    result += int_to_bytes(output_size_bytes)

    result += str_to_bytes(dtype, 4)

    result += str_to_bytes(entry_func, MAX_LENGTH_ENTRY_FUNC_NAME)

    result += str_to_bytes(model_name, MAX_LENGTH_MODEL_NAME)

    assert len(result) == 160, 'Wrong struct size'

    return result


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
        self.selector = selectors.DefaultSelector()
        super().__init__()

    @classmethod
    def from_argparse(cls, args):
        return cls(
            port=args.port,
            baudrate=args.baudrate,
            endianness=args.endianness
        )

    def initialize_client(self) -> bool:
        self.connection = serial.Serial(self.port, self.baudrate)
        self.selector.register(
            self.connection,
            selectors.EVENT_READ | selectors.EVENT_WRITE,
            self.receive_data
        )
        return True

    def wait_for_activity(self) -> List[Tuple['ServerStatus', Any]]:
        events = self.selector.select(timeout=1)
        results = []
        for key, mask in events:
            if mask & selectors.EVENT_READ:
                callback = key.data
                code, data = callback(key.fileobj, mask)
                results.append((code, data))
        if len(results) == 0:
            return [(ServerStatus.NOTHING, None)]
        return results

    def send_data(self, data: bytes) -> bool:
        length = (len(data)).to_bytes(4, self.endianness, signed=False)
        packet = length + data
        self.connection.write(packet)

    def receive_data(self) -> Tuple['ServerStatus', Any]:
        if not self.connection.is_open:
            return ServerStatus.CLIENT_DISCONNECTED, None
        length = int.from_bytes(
            self.connection.read(4),
            byteorder=self.endianness,
            signed=False
        )
        packet = self.connection.read(length)
        return ServerStatus.DATA_READY, packet

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
            for status, data in self.wait_for_activity():
                if status == ServerStatus.DATA_READY:
                    if len(data) != 1:
                        # this should not happen
                        # TODO handle this scenario
                        self.log.error('There are more messages than expected')
                        return False, None
                    typ, dat = self.parse_message(data[0])
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

    def upload_io_specification(self, path: Path) -> bool:
        self.log.debug('Uploading io specification')
        with open(path, 'rb') as io_spec_file:
            io_spec = json.load(io_spec_file)
        data = io_spec_to_iree_model_spec(io_spec, '', '')

        self.send_message(MessageType.IOSPEC, data)
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
