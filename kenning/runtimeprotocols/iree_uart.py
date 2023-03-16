# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
UART-based inference communication protocol.
"""

from typing import Any, Tuple, List, Dict
from pathlib import Path
import re
import json
import numpy as np

from kenning.core.runtimeprotocol import MessageType
from kenning.core.measurements import Measurements
from kenning.runtimeprotocols.uart import UARTProtocol


# model constraints
MAX_MODEL_INPUT_NUM = 2
MAX_MODEL_INPUT_DIM = 4
MAX_MODEL_OUTPUTS = 12
MAX_LENGTH_ENTRY_FUNC_NAME = 20
MAX_LENGTH_MODEL_NAME = 20


def _io_spec_to_iree_model_struct(
        io_spec: Dict[str, Any],
        entry_func: str = 'module.main',
        model_name: str = 'module',
        byteorder: str = 'little') -> bytes:
    """
    Method used to convert IO spec in JSON form into struct that can be easily
    parsed by bare-metal IREE runtime

    Parameters
    ----------
    io_spec : Dict[str, Any]
        Input IO spec
    entry_func : str
        Name of the entry function of the IREE module
    model_name : str
        Name of the model
    byteorder : str
        Byteorder of the struct (either 'little' or 'big')

    Returns
    -------
    bytes :
        IO spec struct
    """
    input_shape = [inp['shape'] for inp in io_spec['input']]
    output_length = [
        int(np.prod(outp['shape'])) for outp in io_spec['output']
    ]
    dtype = io_spec['input'][0]['dtype']

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
        a_pad = np.pad(
            a_np,
            [(0, shape[i] - a_np.shape[i]) for i in range(a_np.ndim)]
        )
        return a_pad.astype(np.uint32).tobytes('C')

    def str_to_bytes(s: str, length: int) -> bytes:
        return s.ljust(length, '\0')[:length].encode('ascii')

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


def _parse_iree_stats(data: bytes) -> Dict[str, int]:
    """
    Method used to parse stats sent by bare-meta IREE runtime

    Parameters
    ----------
    data : bytes
        Byte array with stats sent by runtime

    Returns
    -------
    Dict[str, int] :
        Parsed stats
    """
    stats = np.frombuffer(data, dtype=np.uint32, count=6)
    stats_json = {
        'host_bytes_peak': stats[0],
        'host_bytes_allocated': stats[1],
        'host_bytes_freed': stats[2],
        'device_bytes_peak': stats[3],
        'device_bytes_allocated': stats[4],
        'device_bytes_freed': stats[5],
    }
    return stats_json


class IREEUARTProtocol(UARTProtocol):

    def upload_io_specification(self, path: Path) -> bool:
        self.log.debug('Uploading io specification')
        with open(path, 'rb') as io_spec_file:
            io_spec = json.load(io_spec_file)

        data = _io_spec_to_iree_model_struct(io_spec)

        self.send_message(MessageType.IOSPEC, data)
        return self.receive_confirmation()[0]

    def download_statistics(self) -> 'Measurements':
        self.send_message(MessageType.STATS)
        status, data = self.receive_confirmation()
        measurements = Measurements()
        if status and isinstance(data, bytes) and len(data) > 0:
            measurements += _parse_iree_stats(data)
        return measurements
