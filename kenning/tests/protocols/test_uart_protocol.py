# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import json
import multiprocessing
import os
import selectors
import struct
import time
from pathlib import Path
from random import choices, randint
from string import ascii_lowercase
from typing import Any, Dict, Literal, Tuple

import numpy as np
import pytest
import serial

from kenning.core.model import ModelWrapper
from kenning.core.protocol import ServerAction
from kenning.interfaces.io_spec_serializer import IOSpecSerializer
from kenning.protocols.message import (
    FlagName,
    Flags,
    FlowControlFlags,
    Message,
    MessageType,
)
from kenning.protocols.uart import (
    BARE_METAL_IREE_ALLOCATION_STATS_SIZE,
    RUNTIME_STAT_NAME_MAX_LEN,
    UARTProtocol,
    _parse_stats,
)
from kenning.tests.conftest import get_tmp_path
from kenning.tests.protocols.test_core_protocol import (
    TestCoreProtocol,
)
from kenning.utils.class_loader import get_all_subclasses
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import ResourceURI

MODEL_WRAPPER_SUBCLASSES = get_all_subclasses(
    "kenning.modelwrappers", ModelWrapper, raise_exception=True
)
MODEL_WRAPPER_SUBCLASSES_WITH_IO_SPEC = [
    modelwrapper_cls
    for modelwrapper_cls in MODEL_WRAPPER_SUBCLASSES
    if hasattr(modelwrapper_cls, "pretrained_model_uri")
    and modelwrapper_cls.pretrained_model_uri is not None
    and not modelwrapper_cls.pretrained_model_uri.startswith("hf://")
]


@pytest.fixture
def valid_io_spec() -> Dict[str, Any]:
    modelwrapper_cls = MODEL_WRAPPER_SUBCLASSES_WITH_IO_SPEC[0]
    valid_io_spec_path = ResourceURI(
        f"{modelwrapper_cls.pretrained_model_uri}.json"
    )
    with open(valid_io_spec_path, "r") as io_spec_f:
        io_spec = json.load(io_spec_f)

    return io_spec


@pytest.fixture
def valid_iree_stats() -> bytes:
    stats = np.random.randint(
        np.iinfo(np.uint32).min, np.iinfo(np.uint32).max, 6, np.uint32
    )
    return stats.tobytes()


@pytest.fixture
def valid_generic_stats() -> bytes:
    stats_names = [
        "".join(
            choices(
                ascii_lowercase + "_",
                k=randint(1, RUNTIME_STAT_NAME_MAX_LEN - 1),
            )
        )
        for _ in range(4)
    ]
    stats_values = np.random.randint(
        np.iinfo(np.uint64).min, np.iinfo(np.uint64).max, 4, np.uint64
    )

    stats = b""
    for name, value in zip(stats_names, stats_values):
        struct.pack(f"{RUNTIME_STAT_NAME_MAX_LEN}sQQ", name.encode(), 0, value)

    return stats


def mock_serial() -> Tuple[Path, Path]:
    class SerialMock:
        def __init__(self, port: Tuple[Path, Path], *args, **kwargs):
            self.fifo_in = open(port[0], "wb+", 0)
            self.fifo_out = open(port[1], "wb+", 0)
            os.set_blocking(self.fifo_in.fileno(), False)
            os.set_blocking(self.fifo_out.fileno(), False)
            self.is_open = True

        def read(self, size: int = -1):
            data = self.fifo_in.read(size)
            KLogger.debug(f"SerialMock read {data}")
            return data

        def write(self, data: bytes):
            KLogger.debug(f"SerialMock write {bytes}")
            return self.fifo_out.write(data)

        def fileno(self):
            return self.fifo_in.fileno()

        def close(self):
            self.fifo_in.close()
            self.fifo_out.close()
            self.is_open = False

    serial.Serial = SerialMock

    fifo_in = get_tmp_path(".uart_in")
    fifo_out = get_tmp_path(".uart_out")
    os.mkfifo(str(fifo_in))
    os.mkfifo(str(fifo_out))

    return fifo_in, fifo_out


class TestParseAllocationStats:
    def test_parse_valid_iree_stats(self, valid_iree_stats: bytes):
        stats_json = _parse_stats(valid_iree_stats)["allocations"]

        assert len(stats_json) == BARE_METAL_IREE_ALLOCATION_STATS_SIZE // 4
        assert all(
            [isinstance(stat_name, str) for stat_name in stats_json.keys()]
        )
        assert all([isinstance(stat, int) for stat in stats_json.values()])
        assert json.loads(json.dumps(stats_json)) == stats_json

    def test_parse_valid_generic_stats(self, valid_generic_stats: bytes):
        stats_json = _parse_stats(valid_generic_stats)

        assert len(stats_json) == len(valid_generic_stats) / (
            RUNTIME_STAT_NAME_MAX_LEN + 8
        )
        assert all(
            [isinstance(stat_name, str) for stat_name in stats_json.keys()]
        )
        assert all([isinstance(stat, int) for stat in stats_json.values()])
        assert json.loads(json.dumps(stats_json)) == stats_json

    @pytest.mark.parametrize(
        "invalid_size",
        [
            BARE_METAL_IREE_ALLOCATION_STATS_SIZE - 4,
            BARE_METAL_IREE_ALLOCATION_STATS_SIZE - 1,
            BARE_METAL_IREE_ALLOCATION_STATS_SIZE + 1,
            BARE_METAL_IREE_ALLOCATION_STATS_SIZE + 4,
            BARE_METAL_IREE_ALLOCATION_STATS_SIZE + 100,
        ],
    )
    def test_parse_stats_with_invalid_size(
        self, valid_iree_stats: bytes, invalid_size: int
    ):
        invalid_stats = (
            valid_iree_stats * (invalid_size // len(valid_iree_stats))
            + valid_iree_stats[: invalid_size % len(valid_iree_stats)]
        )

        with pytest.raises(ValueError):
            _ = _parse_stats(invalid_stats)

    @pytest.mark.parametrize(
        "invalid_string",
        [
            b"abc\xffd",
            b"\xff\xff",
            b"\xab\xcd",
        ],
    )
    def test_parse_invalid_generic_stats(
        self, valid_generic_stats: bytes, invalid_string: bytes
    ):
        invalid_stats = (
            invalid_string + valid_generic_stats[len(invalid_string) :]
        )

        with pytest.raises(ValueError):
            _ = _parse_stats(invalid_stats)


class TestUARTProtocol(TestCoreProtocol):
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
        UARTProtocol
            Initialized UART protocol client.
        """
        client = self.init_protocol()
        client.connection = serial.Serial(
            client.port, client.baudrate, timeout=0
        )
        client.selector.register(
            client.connection,
            selectors.EVENT_READ | selectors.EVENT_WRITE,
            client.receive_data,
        )
        client.start()
        yield client
        client.stop()
        client.connection.close()

    def mock_recv_message(self, message_type: MessageType):
        def recv_message(queue: multiprocessing.Queue):
            data = b""
            time.sleep(0.1)
            # wait for msg
            with open(self.port_out, "rb", 0) as serial_f:
                os.set_blocking(serial_f.fileno(), False)
                while True:
                    read = serial_f.read()
                    if read is not None:
                        data += read
                    message, data_parsed, checksum_valid = Message.from_bytes(
                        data
                    )
                    if message is not None:
                        break
                    time.sleep(0.01)

                queue.put(message)

            if message.message_type == message_type:
                response = Message(
                    message_type,
                    None,
                    FlowControlFlags.TRANSMISSION,
                    Flags(
                        {
                            FlagName.SUCCESS: True,
                            FlagName.IS_ZEPHYR: True,
                            FlagName.FIRST: True,
                            FlagName.LAST: True,
                        }
                    ),
                )
            else:
                response = Message(
                    message_type,
                    None,
                    FlowControlFlags.TRANSMISSION,
                    Flags(
                        {
                            FlagName.FAIL: True,
                            FlagName.IS_ZEPHYR: True,
                            FlagName.FIRST: True,
                            FlagName.LAST: True,
                        }
                    ),
                )
            with open(self.port_in, "wb", 0) as serial_f:
                serial_f.write(response.to_bytes())

        return recv_message

    def mock_send_response_to_request(
        self, message_type: MessageType, payload: bytes = b""
    ):
        def send_message():
            data = b""
            # wait for msg
            with open(self.port_out, "rb", 0) as serial_f:
                os.set_blocking(serial_f.fileno(), False)
                while True:
                    read = serial_f.read()
                    if read is not None:
                        data += read
                    message, data_parsed, checksum_valid = Message.from_bytes(
                        data
                    )
                    if message is not None:
                        break
                    time.sleep(0.01)

            # send stats
            if message.message_type == message_type:
                response = Message(
                    message.message_type,
                    payload,
                    FlowControlFlags.TRANSMISSION,
                    Flags(
                        {
                            FlagName.SUCCESS: True,
                            FlagName.FIRST: True,
                            FlagName.LAST: True,
                        }
                    ),
                )
            else:
                response = Message(
                    message.message_type,
                    ServerAction.COMPUTING_STATISTICS.to_bytes(),
                    FlowControlFlags.TRANSMISSION,
                    Flags(
                        {
                            FlagName.FAIL: True,
                            FlagName.FIRST: True,
                            FlagName.LAST: True,
                        }
                    ),
                )

            with open(self.port_in, "wb", 0) as serial_f:
                serial_f.write(response.to_bytes())

        return send_message

    def test_initialize_client_disconnect(self, client: UARTProtocol):
        """
        Test disconnect client method.
        """
        queue = multiprocessing.Queue()
        thread_recv = multiprocessing.Process(
            target=self.mock_recv_message(MessageType.PING),
            args=(queue,),
        )
        thread_recv.start()
        ret = client.initialize_client()
        thread_recv.join()
        assert ret
        assert queue.qsize() == 1
        message = queue.get()
        assert isinstance(message, Message)
        assert message.message_type == MessageType.PING
        assert message.payload == b""
        assert message.flags.success
        thread_recv = multiprocessing.Process(
            target=self.mock_recv_message(MessageType.PING),
            args=(queue,),
        )
        thread_recv.start()
        client.disconnect()
        thread_recv.join()
        assert queue.qsize() == 1
        message = queue.get()
        assert isinstance(message, Message)
        assert message.message_type == MessageType.PING
        assert message.payload == b""
        assert message.flags.fail
        assert client.send_message(Message(MessageType.MODEL)) is False
        assert client.connection.is_open is False

    @pytest.mark.parametrize(
        "message_type", [MessageType.DATA, MessageType.IO_SPEC]
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
        # We need to stop the KenningProtocol receiver thread, because it is
        # calling 'receive_message' too, which causes a race condition.
        client.stop()
        message = client.receive_message(timeout=1)
        assert message is None

        # send data
        self.port_in.write_bytes(
            Message(message_type, random_byte_data).to_bytes()
        )
        message = client.receive_message(timeout=1)
        assert (
            message.payload == random_byte_data
            and message.message_type == message_type
        )

    def test_receive_empty_message(self, client: UARTProtocol):
        """
        Test client receive_message method.
        """
        self.port_in.write_bytes(b"")
        message = client.receive_message(timeout=1)
        assert message is None

    @pytest.mark.parametrize(
        "message_type",
        [MessageType.STATUS, MessageType.IO_SPEC, MessageType.DATA],
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
        with open(self.port_out, "rb", 0) as serial_f:
            os.set_blocking(serial_f.fileno(), False)
            received_message, bytes_read, checksum_valid = Message.from_bytes(
                serial_f.read()
            )

        assert bytes_read == len(message.to_bytes())
        assert received_message is not None
        assert received_message.message_type == message_type
        assert received_message.payload == random_byte_data

    def test_send_empty_message(self, client: UARTProtocol):
        """
        Test client send_message method.
        """

        class EmptyMessage:
            def to_bytes(self):
                return b""

        # send data
        client.send_message(EmptyMessage())
        with open(self.port_out, "rb", 0) as serial_f:
            os.set_blocking(serial_f.fileno(), False)
            bytes_read = serial_f.read()

        assert bytes_read is None

    def test_send_data(self, client: UARTProtocol, random_byte_data: bytes):
        """
        Test client send_data method.
        """
        client.send_data(random_byte_data)

        with open(self.port_out, "rb", 0) as serial_f:
            os.set_blocking(serial_f.fileno(), False)
            received_data = serial_f.read()

        assert random_byte_data == received_data

    def test_receive_data(self, client: UARTProtocol, random_byte_data: bytes):
        """
        Test client send_data method.
        """
        # We need to stop the KenningProtocol receiver thread, because it is
        # calling 'receive_message', which calls 'receive_data', which
        # causes a race condition.
        client.stop()
        with open(self.port_in, "wb", 0) as serial_f:
            serial_f.write(random_byte_data)

        received_data = client.receive_data(None, None)

        assert random_byte_data == received_data

    def test_upload_input(self, client: UARTProtocol, random_byte_data: bytes):
        """
        Test client upload_input method.
        """
        queue = multiprocessing.Queue()
        thread_recv = multiprocessing.Process(
            target=self.mock_recv_message(MessageType.DATA),
            args=(queue,),
        )
        thread_recv.start()

        ret = client.upload_input(random_byte_data)

        thread_recv.join()
        assert ret
        assert queue.qsize() == 1
        message = queue.get()
        assert isinstance(message, Message)
        assert message.message_type == MessageType.DATA
        assert message.payload == random_byte_data

    def test_upload_model(self, client: UARTProtocol, random_byte_data: bytes):
        """
        Test client upload_input method.
        """
        queue = multiprocessing.Queue()
        thread_recv = multiprocessing.Process(
            target=self.mock_recv_message(MessageType.MODEL),
            args=(queue,),
        )
        thread_recv.start()

        model_path = get_tmp_path()
        model_path.write_bytes(random_byte_data)
        ret = client.upload_model(model_path)

        thread_recv.join()
        assert ret
        assert queue.qsize() == 1
        message = queue.get()
        assert isinstance(message, Message)
        assert message.message_type == MessageType.MODEL
        assert message.payload == random_byte_data

    def test_upload_io_specification(
        self, client: UARTProtocol, valid_io_spec: Dict[str, Any]
    ):
        """
        Test client upload_io_specification method.
        """

        def io_spec_to_struct_mock(
            io_spec: Dict[str, Any],
            entry_func: str = "module.main",
            model_name: str = "module",
            byteorder: Literal["little", "big"] = "little",
        ) -> bytes:
            return b"\x05\x04\x03\x02\x01"

        IOSpecSerializer.io_spec_to_struct = io_spec_to_struct_mock

        queue = multiprocessing.Queue()
        thread_recv = multiprocessing.Process(
            target=self.mock_recv_message(MessageType.IO_SPEC),
            args=(queue,),
        )
        thread_recv.start()
        io_spec_path = get_tmp_path()
        io_spec_path.write_text(json.dumps(valid_io_spec))

        ret = client.upload_io_specification(io_spec_path)

        thread_recv.join()
        assert ret
        assert queue.qsize() == 1
        message = queue.get()
        assert isinstance(message, Message)
        assert message.message_type == MessageType.IO_SPEC
        assert message.payload == io_spec_to_struct_mock(valid_io_spec)

    def test_request_processing(self, client: UARTProtocol):
        """
        Test client request_processing method.
        """
        queue = multiprocessing.Queue()
        thread_recv = multiprocessing.Process(
            target=self.mock_recv_message(MessageType.PROCESS),
            args=(queue,),
        )
        thread_recv.start()
        ret = client.request_processing()

        thread_recv.join()
        assert ret
        assert queue.qsize() == 1
        message = queue.get()
        assert isinstance(message, Message)
        assert message.message_type == MessageType.PROCESS
        assert message.payload == b""

    def test_download_statistics(
        self, client: UARTProtocol, valid_iree_stats: bytes
    ):
        """
        Test client download_statistics method.
        """
        thread_send = multiprocessing.Process(
            target=self.mock_send_response_to_request(
                MessageType.STATS, valid_iree_stats
            )
        )

        thread_send.start()
        statistics = client.download_statistics(final=True)

        thread_send.join()
        assert _parse_stats(valid_iree_stats) == statistics.data
