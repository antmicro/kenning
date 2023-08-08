# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import json
import multiprocessing
import os
import time
from contextlib import nullcontext as does_not_raise
from multiprocessing.managers import ListProxy
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import serial

from kenning.core.model import ModelWrapper
from kenning.core.runtimeprotocol import Message, MessageType, ServerStatus
from kenning.runtimeprotocols.uart import (
    ALLOCATION_STATS_SIZE,
    MAX_LENGTH_ENTRY_FUNC_NAME,
    MAX_LENGTH_MODEL_NAME,
    MAX_MODEL_INPUT_DIM,
    MAX_MODEL_INPUT_NUM,
    MAX_MODEL_OUTPUTS,
    MODEL_STRUCT_SIZE,
    UARTProtocol,
    _io_spec_to_struct,
    _parse_allocation_stats,
)
from kenning.tests.conftest import get_tmp_path
from kenning.tests.runtimeprotocols.test_core_protocol import (
    TestCoreRuntimeProtocol,
)
from kenning.utils.class_loader import get_all_subclasses
from kenning.utils.logger import get_logger
from kenning.utils.resource_manager import ResourceURI

MODEL_WRAPPER_SUBCLASSES = get_all_subclasses(
    'kenning.modelwrappers', ModelWrapper, raise_exception=True
)
MODEL_WRAPPER_SUBCLASSES_WITH_IO_SPEC = [
    modelwrapper_cls
    for modelwrapper_cls in MODEL_WRAPPER_SUBCLASSES
    if (
        hasattr(modelwrapper_cls, 'pretrained_model_uri')
        and ResourceURI(
            f'{modelwrapper_cls.pretrained_model_uri}.json'
        ).exists()
    )
]


@pytest.fixture
def valid_io_spec() -> Dict[str, Any]:
    modelwrapper_cls = MODEL_WRAPPER_SUBCLASSES_WITH_IO_SPEC[0]
    valid_io_spec_path = ResourceURI(
        f'{modelwrapper_cls.pretrained_model_uri}.json'
    )
    with open(valid_io_spec_path, 'r') as io_spec_f:
        io_spec = json.load(io_spec_f)

    return io_spec


@pytest.fixture
def valid_stats() -> bytes:
    stats = np.random.randint(
        np.iinfo(np.uint32).min, np.iinfo(np.uint32).max, 6, np.uint32
    )
    return stats.tobytes()


def mock_serial() -> Path:
    class SerialMock:
        def __init__(self, port: Tuple[Path, Path], *args, **kwargs):
            self.fifo_in = open(port[0], 'wb+', 0)
            self.fifo_out = open(port[1], 'wb+', 0)
            os.set_blocking(self.fifo_in.fileno(), False)
            os.set_blocking(self.fifo_out.fileno(), False)
            self.is_open = True
            self.log = get_logger()

        def read(self, size: int = -1):
            data = self.fifo_in.read(size)
            self.log.debug(f'SerialMock read {data}')
            return data

        def write(self, data: bytes):
            self.log.debug(f'SerialMock write {bytes}')
            return self.fifo_out.write(data)

        def fileno(self):
            return self.fifo_in.fileno()

        def close(self):
            self.fifo_in.close()
            self.fifo_out.close()
            self.is_open = False

    serial.Serial = SerialMock

    fifo_in = get_tmp_path('.uart_in')
    fifo_out = get_tmp_path('.uart_out')
    os.mkfifo(str(fifo_in))
    os.mkfifo(str(fifo_out))

    return fifo_in, fifo_out


class TestIOSpecToStruct:
    @pytest.mark.parametrize(
        'io_spec_path',
        [
            ResourceURI(f'{modelwrapper_cls.pretrained_model_uri}.json')
            for modelwrapper_cls in MODEL_WRAPPER_SUBCLASSES_WITH_IO_SPEC
        ],
        ids=[
            modelwrapper_cls.__name__
            for modelwrapper_cls in MODEL_WRAPPER_SUBCLASSES_WITH_IO_SPEC
        ],
    )
    def test_parse_valid_io_spec(self, io_spec_path: ResourceURI):
        with open(io_spec_path, 'r') as io_spec_f:
            io_spec = json.load(io_spec_f)

        struct = _io_spec_to_struct(io_spec)

        assert len(struct) == MODEL_STRUCT_SIZE

    @pytest.mark.parametrize(
        'dtype,expectation',
        [
            ('f32', does_not_raise()),
            ('u8', does_not_raise()),
            ('i16', does_not_raise()),
            ('f', pytest.raises(ValueError)),
            ('f31', pytest.raises(ValueError)),
            ('f0', pytest.raises(ValueError)),
            ('u9', pytest.raises(ValueError)),
        ],
    )
    def test_parse_io_spec_with_different_dtypes(
        self, valid_io_spec: Dict[str, Any], dtype: str, expectation
    ):
        valid_io_spec['input'][0]['dtype'] = dtype

        with expectation:
            struct = _io_spec_to_struct(valid_io_spec)

            assert len(struct) == MODEL_STRUCT_SIZE

    @pytest.mark.parametrize(
        'inputs_num,expectation',
        [
            (1, does_not_raise()),
            (MAX_MODEL_INPUT_NUM, does_not_raise()),
            (MAX_MODEL_INPUT_NUM + 1, pytest.raises(ValueError)),
            (MAX_MODEL_INPUT_NUM + 100, pytest.raises(ValueError)),
        ],
    )
    def test_parse_io_spec_with_different_inputs_num(
        self, valid_io_spec: Dict[str, Any], inputs_num, expectation
    ):
        valid_io_spec['input'] = inputs_num * [valid_io_spec['input'][0]]

        with expectation:
            struct = _io_spec_to_struct(valid_io_spec)

            assert len(struct) == MODEL_STRUCT_SIZE

    @pytest.mark.parametrize(
        'input_dim,expectation',
        [
            (1, does_not_raise()),
            (MAX_MODEL_INPUT_DIM, does_not_raise()),
            (MAX_MODEL_INPUT_DIM + 1, pytest.raises(ValueError)),
            (MAX_MODEL_INPUT_DIM + 100, pytest.raises(ValueError)),
        ],
    )
    def test_parse_io_spec_with_different_input_dim(
        self, valid_io_spec: Dict[str, Any], input_dim, expectation
    ):
        dims = (input_dim - len(valid_io_spec['input'][0]['shape'])) * [
            1,
        ]
        valid_io_spec['input'][0]['shape'] = (
            *valid_io_spec['input'][0]['shape'],
            *dims,
        )

        with expectation:
            struct = _io_spec_to_struct(valid_io_spec)

            assert len(struct) == MODEL_STRUCT_SIZE

    @pytest.mark.parametrize(
        'outputs_num,expectation',
        [
            (1, does_not_raise()),
            (MAX_MODEL_OUTPUTS, does_not_raise()),
            (MAX_MODEL_OUTPUTS + 1, pytest.raises(ValueError)),
            (MAX_MODEL_OUTPUTS + 100, pytest.raises(ValueError)),
        ],
    )
    def test_parse_io_spec_with_different_outputs_num(
        self, valid_io_spec: Dict[str, Any], outputs_num, expectation
    ):
        valid_io_spec['output'] = outputs_num * [valid_io_spec['output'][0]]

        with expectation:
            struct = _io_spec_to_struct(valid_io_spec)

            assert len(struct) == MODEL_STRUCT_SIZE

    @pytest.mark.parametrize(
        'entry_func_name_len,expectation',
        [
            (1, does_not_raise()),
            (MAX_LENGTH_ENTRY_FUNC_NAME, does_not_raise()),
            (MAX_LENGTH_ENTRY_FUNC_NAME + 1, pytest.raises(ValueError)),
            (MAX_LENGTH_ENTRY_FUNC_NAME + 100, pytest.raises(ValueError)),
        ],
    )
    def test_parse_io_spec_with_different_entry_func(
        self, valid_io_spec: Dict[str, Any], entry_func_name_len, expectation
    ):
        entry_func = 'a' * entry_func_name_len

        with expectation:
            struct = _io_spec_to_struct(valid_io_spec, entry_func=entry_func)

            assert len(struct) == MODEL_STRUCT_SIZE

    @pytest.mark.parametrize(
        'model_name_len,expectation',
        [
            (1, does_not_raise()),
            (MAX_LENGTH_MODEL_NAME, does_not_raise()),
            (MAX_LENGTH_MODEL_NAME + 1, pytest.raises(ValueError)),
            (MAX_LENGTH_MODEL_NAME + 100, pytest.raises(ValueError)),
        ],
    )
    def test_parse_io_spec_with_different_model_name(
        self, valid_io_spec: Dict[str, Any], model_name_len, expectation
    ):
        model_name = 'a' * model_name_len

        with expectation:
            struct = _io_spec_to_struct(valid_io_spec, model_name=model_name)

            assert len(struct) == MODEL_STRUCT_SIZE


class TestParseAllocationStats:
    def test_parse_valid_stats(self, valid_stats: bytes):
        stats_json = _parse_allocation_stats(valid_stats)

        assert len(stats_json) == ALLOCATION_STATS_SIZE // 4
        assert all(
            [isinstance(stat_name, str) for stat_name in stats_json.keys()]
        )
        assert all([isinstance(stat, int) for stat in stats_json.values()])
        assert json.loads(json.dumps(stats_json)) == stats_json

    @pytest.mark.parametrize(
        'invalid_size',
        [
            0,
            ALLOCATION_STATS_SIZE - 4,
            ALLOCATION_STATS_SIZE - 1,
            ALLOCATION_STATS_SIZE + 1,
            ALLOCATION_STATS_SIZE + 4,
            ALLOCATION_STATS_SIZE + 100,
        ],
    )
    def test_parse_stats_with_invalid_size(
        self, valid_stats: bytes, invalid_size: int
    ):
        invalid_stats = [
            valid_stats[i % len(valid_stats)] for i in range(invalid_size)
        ]

        with pytest.raises(ValueError):
            _ = _parse_allocation_stats(invalid_stats)


class TestUARTProtocol(TestCoreRuntimeProtocol):
    port = mock_serial()
    port_in = port[0]
    port_out = port[1]

    def init_protocol(self):
        return UARTProtocol(port=self.port)

    @pytest.fixture
    def client(self):
        """
        Initialize client.

        Returns
        -------
        UARTProtocol :
            Initialized UART protocol client.
        """
        client = self.init_protocol()
        if client.initialize_client() is False:
            pytest.fail('Client initialization failed')

        yield client

        client.disconnect()

    def mock_recv_message(self, message_type: MessageType):
        def recv_message(shared_list: ListProxy):
            data = b''
            time.sleep(0.1)
            # wait for msg
            with open(self.port_out, 'rb', 0) as serial_f:
                os.set_blocking(serial_f.fileno(), False)
                while True:
                    read = serial_f.read()
                    if read is not None:
                        data += read
                    message, data_parsed = Message.from_bytes(data)
                    if message is not None:
                        break
                    time.sleep(0.01)

                shared_list.append(message)

            if message.message_type == message_type:
                response = Message(MessageType.OK)
            else:
                response = Message(MessageType.ERROR)

            with open(self.port_in, 'wb', 0) as serial_f:
                serial_f.write(response.to_bytes())

        return recv_message

    def mock_send_response(
        self, message_type: MessageType, payload: bytes = b''
    ):
        def send_message():
            data = b''
            # wait for msg
            with open(self.port_out, 'rb', 0) as serial_f:
                os.set_blocking(serial_f.fileno(), False)
                while True:
                    read = serial_f.read()
                    if read is not None:
                        data += read
                    message, data_parsed = Message.from_bytes(data)
                    if message is not None:
                        break
                    time.sleep(0.01)

            # send stats
            if message.message_type == message_type:
                response = Message(MessageType.OK, payload)
            else:
                response = Message(MessageType.ERROR)

            with open(self.port_in, 'wb', 0) as serial_f:
                serial_f.write(response.to_bytes())

        return send_message

    def test_initialize_client(self):
        """
        Test the `initialize_client method.
        """
        client = self.init_protocol()
        assert client.initialize_client()
        client.disconnect()

    def test_disconnect(self, client: UARTProtocol):
        """
        Test disconnect client method.
        """
        client = self.init_protocol()
        assert client.initialize_client()

        client.disconnect()
        assert client.send_message(Message(MessageType.OK)) is False
        assert client.connection.is_open is False

    @pytest.mark.parametrize(
        'message_type', [MessageType.OK, MessageType.ERROR]
    )
    def test_receive_message(
        self,
        client: UARTProtocol,
        message_type: MessageType,
        random_byte_data: bytes,
    ):
        """
        Test client receive_message method.
        """

        status, message = client.receive_message(timeout=1)
        assert status == ServerStatus.NOTHING
        assert message is None

        # send data
        self.port_in.write_bytes(
            Message(message_type, random_byte_data).to_bytes()
        )
        status, message = client.receive_message(timeout=1)
        assert status == ServerStatus.DATA_READY
        assert (
            message.payload == random_byte_data
            and message.message_type == message_type
        )

    def test_receive_empty_message(self, client: UARTProtocol):
        """
        Test client receive_message method.
        """

        self.port_in.write_bytes(b'')
        status, message = client.receive_message(timeout=1)
        assert status == ServerStatus.NOTHING
        assert message is None

    @pytest.mark.parametrize(
        'message_type', [MessageType.OK, MessageType.ERROR, MessageType.DATA]
    )
    def test_send_message(
        self,
        client: UARTProtocol,
        message_type: MessageType,
        random_byte_data: bytes,
    ):
        """
        Test client send_message method.
        """

        message = Message(message_type, random_byte_data)

        # send data
        client.send_message(message)
        with open(self.port_out, 'rb', 0) as serial_f:
            os.set_blocking(serial_f.fileno(), False)
            received_message, bytes_read = Message.from_bytes(serial_f.read())

        assert bytes_read == len(message.to_bytes())
        assert received_message is not None
        assert received_message.message_type == message_type
        assert received_message.payload == random_byte_data

    def test_send_empty_message(self, client: UARTProtocol):
        """
        Test client send_message method.
        """

        class EmptyMessage(object):
            def to_bytes(self):
                return b''

        # send data
        client.send_message(EmptyMessage())
        with open(self.port_out, 'rb', 0) as serial_f:
            os.set_blocking(serial_f.fileno(), False)
            bytes_read = serial_f.read()

        assert bytes_read is None

    def test_send_data(self, client: UARTProtocol, random_byte_data: bytes):
        """
        Test client send_data method.
        """

        client.send_data(random_byte_data)

        with open(self.port_out, 'rb', 0) as serial_f:
            os.set_blocking(serial_f.fileno(), False)
            received_data = serial_f.read()

        assert random_byte_data == received_data

    def test_receive_data(self, client: UARTProtocol, random_byte_data: bytes):
        """
        Test client send_data method.
        """

        with open(self.port_in, 'wb', 0) as serial_f:
            serial_f.write(random_byte_data)

        status, received_data = client.receive_data(None, None)

        assert status == ServerStatus.DATA_READY
        assert random_byte_data == received_data

    def test_request_success(self, client: UARTProtocol):
        """
        Test client request success.
        """

        with open(self.port_in, 'wb', 0) as serial_f:
            serial_f.write(Message(MessageType.OK).to_bytes())

        status, data = client.receive_data(None, None)

        assert status == ServerStatus.DATA_READY
        message = client.parse_message(data)
        assert message.message_type == MessageType.OK
        assert message.payload == b''

    def test_request_failure(self, client: UARTProtocol):
        """
        Test client request failure.
        """

        with open(self.port_in, 'wb', 0) as serial_f:
            serial_f.write(Message(MessageType.ERROR).to_bytes())

        status, data = client.receive_data(None, None)

        assert status == ServerStatus.DATA_READY
        message = client.parse_message(data)
        assert message.message_type == MessageType.ERROR
        assert message.payload == b''

    def test_upload_input(self, client: UARTProtocol, random_byte_data: bytes):
        """
        Test client upload_input method.
        """

        shared_list = (multiprocessing.Manager()).list()
        thread_recv = multiprocessing.Process(
            target=self.mock_recv_message(MessageType.DATA),
            args=(shared_list,),
        )
        thread_recv.start()

        ret = client.upload_input(random_byte_data)

        thread_recv.join()

        assert ret
        assert len(shared_list) == 1
        assert isinstance(shared_list[0], Message)
        assert shared_list[0].message_type == MessageType.DATA
        assert shared_list[0].payload == random_byte_data

    def test_upload_model(self, client: UARTProtocol, random_byte_data: bytes):
        """
        Test client upload_input method.
        """

        shared_list = (multiprocessing.Manager()).list()
        thread_recv = multiprocessing.Process(
            target=self.mock_recv_message(MessageType.MODEL),
            args=(shared_list,),
        )
        thread_recv.start()

        model_path = get_tmp_path()
        model_path.write_bytes(random_byte_data)

        ret = client.upload_model(model_path)

        thread_recv.join()

        assert ret
        assert len(shared_list) == 1
        assert isinstance(shared_list[0], Message)
        assert shared_list[0].message_type == MessageType.MODEL
        assert shared_list[0].payload == random_byte_data

    def test_upload_io_specification(
        self, client: UARTProtocol, valid_io_spec: Dict[str, Any]
    ):
        """
        Test client upload_io_specification method.
        """

        shared_list = (multiprocessing.Manager()).list()
        thread_recv = multiprocessing.Process(
            target=self.mock_recv_message(MessageType.IOSPEC),
            args=(shared_list,),
        )
        thread_recv.start()

        io_spec_path = get_tmp_path()
        io_spec_path.write_text(json.dumps(valid_io_spec))

        ret = client.upload_io_specification(io_spec_path)

        thread_recv.join()

        assert ret
        assert len(shared_list) == 1
        assert isinstance(shared_list[0], Message)
        assert shared_list[0].message_type == MessageType.IOSPEC
        assert shared_list[0].payload == _io_spec_to_struct(valid_io_spec)

    def test_request_processing(self, client: UARTProtocol):
        """
        Test client request_processing method.
        """

        shared_list = (multiprocessing.Manager()).list()
        thread_recv = multiprocessing.Process(
            target=self.mock_recv_message(MessageType.PROCESS),
            args=(shared_list,),
        )
        thread_recv.start()

        ret = client.request_processing()

        thread_recv.join()

        assert ret
        assert len(shared_list) == 1
        assert isinstance(shared_list[0], Message)
        assert shared_list[0].message_type == MessageType.PROCESS
        assert shared_list[0].payload == b''

    def test_download_statistics(
        self, client: UARTProtocol, valid_stats: np.ndarray
    ):
        """
        Test client download_statistics method.
        """

        thread_send = multiprocessing.Process(
            target=self.mock_send_response(MessageType.STATS, valid_stats)
        )
        thread_send.start()

        statistics = client.download_statistics()

        thread_send.join()

        assert _parse_allocation_stats(valid_stats) == statistics.data
