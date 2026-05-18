# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import json
import multiprocessing
import struct
import uuid
from pathlib import Path
from random import choices, randint
from string import ascii_lowercase
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from kenning.core.exceptions import ProtocolNotStartedError
from kenning.core.measurements import Measurements
from kenning.core.model import ModelWrapper
from kenning.core.protocol import ServerAction
from kenning.protocols.bytes_based_protocol import TransmissionFlag
from kenning.protocols.message import Message, MessageType
from kenning.protocols.pipe_protocol import PipeProtocol
from kenning.protocols.uart import (
    RUNTIME_STAT_NAME_MAX_LEN,
)
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


def random_path_name():
    return uuid.uuid4().hex


class TestPipeProtocol(TestCoreProtocol):
    pipe_path = random_path_name()

    def init_protocol(self):
        return PipeProtocol(pipe_path=self.pipe_path)

    def test_initialize_server(self):
        """
        Tests the `initialize_server()` method.
        """
        server = self.init_protocol()
        assert server.initialize_server()
        second_server = self.init_protocol()
        assert not second_server.initialize_server()
        server.disconnect()

    def test_initialize_client(self):
        """
        Tests the `initialize_client()` method.
        """
        client = self.init_protocol()
        server = self.init_protocol()
        assert server.initialize_server()
        assert client.initialize_client()
        client.disconnect()
        server.disconnect()

    def test_receive_message(
        self,
        server_and_client: Tuple[PipeProtocol, PipeProtocol],
        random_byte_data: bytes,
    ):
        """
        Tests the `receive_message()` method by sending data.
        """
        server, client = server_and_client
        server.stop()
        message = server.receive_message(timeout=1)
        assert message is None, "Message not received."

        # Send data
        client.send_message(Message(MessageType.OUTPUT, random_byte_data))
        message = server.receive_message(timeout=1)
        assert (
            message.payload == random_byte_data
            and message.message_type == MessageType.OUTPUT
        ), "Received message is incorrect."

    def test_receive_message_send_empty(
        self, server_and_client: Tuple[PipeProtocol, PipeProtocol]
    ):
        """
        Tests the `receive_message()` method by sending empty message.
        """
        server, client = server_and_client
        server.stop()

        # Send empty message
        class EmptyMessage(object):
            def to_bytes(self, verify_checksum: bool):
                return b""

        client.send_message(EmptyMessage())
        message = server.receive_message(timeout=1)
        assert message is None

    def test_send_data(
        self,
        server_and_client: Tuple[PipeProtocol, PipeProtocol],
        random_byte_data: bytes,
    ):
        """
        Tests the `send_data()` method.
        """
        server, client = server_and_client
        server.stop()
        assert client.send_data(random_byte_data)

    def test_receive_data(
        self, server_and_client: Tuple[PipeProtocol, PipeProtocol]
    ):
        """
        Tests the `receive_data()` method with not initialized server.
        """
        server, client = server_and_client
        server.stop()
        server.disconnect()
        with pytest.raises(ProtocolNotStartedError):
            server.receive_data(None)

    def test_receive_data_data_sent(
        self,
        server_and_client: Tuple[PipeProtocol, PipeProtocol],
        random_byte_data: bytes,
    ):
        """
        Tests the `receive_data()` method with data being sent.
        """
        server, client = server_and_client
        server.stop()
        assert client.send_data(random_byte_data)
        received_data = bytearray()

        for _ in random_byte_data:
            received_data += server.receive_data(None)
        assert random_byte_data == received_data

    def test_receive_client_disconnect(
        self, server_and_client: Tuple[PipeProtocol, PipeProtocol]
    ):
        """
        Tests the `receive_data()` method with client being disconnected.
        """
        server, client = server_and_client
        server.stop()

        mock_client_disconnected_callback_call_count = 0

        def mock_client_disconnected_callback():
            nonlocal mock_client_disconnected_callback_call_count
            mock_client_disconnected_callback_call_count += 1

        server.client_disconnected_callback = mock_client_disconnected_callback
        client.disconnect()
        received_data = server.receive_data(None)
        assert received_data is None
        assert 1 == mock_client_disconnected_callback_call_count

    def test_send_message(
        self,
        server_and_client: Tuple[PipeProtocol, PipeProtocol],
        random_byte_data: bytes,
    ):
        """
        Tests the `send_message(Message())` method.
        """
        server, client = server_and_client
        server.stop()
        client.stop()
        assert client.send_message(Message(MessageType.DATA, random_byte_data))
        assert server.send_message(Message(MessageType.DATA, random_byte_data))

        client.disconnect()
        with pytest.raises(ConnectionResetError):
            server.send_message(Message(MessageType.DATA))

    def _receive_request(
        self,
        response_payload: bytes,
        method: str,
        argument: Any,
        message_type: MessageType,
    ) -> bytes:
        def receive(
            pipe_path: str,
            response_payload: bytes,
            server_started_event: multiprocessing.Event,
            queue: multiprocessing.Queue,
        ):
            server = PipeProtocol(pipe_path)
            server.initialize_server()
            server_started_event.set()
            type, message_type, data, flags = server.listen_blocking(
                None, None, None, None
            )
            queue.put(data)
            server.transmit_blocking(
                message_type,
                response_payload,
                [TransmissionFlag.SUCCESS, TransmissionFlag.IS_KENNING],
            )
            server.disconnect()

        queue = multiprocessing.Queue()
        server_started_event = multiprocessing.Event()
        thread = multiprocessing.Process(
            target=receive,
            args=(
                self.pipe_path,
                response_payload,
                server_started_event,
                queue,
            ),
        )
        thread.start()
        server_started_event.wait()
        client = PipeProtocol(self.pipe_path)
        assert client.initialize_client()
        KLogger.debug("Client started for receive event.")
        if argument is not None:
            return_value = getattr(client, method)(argument)
        else:
            return_value = getattr(client, method)()
        client.disconnect()
        thread.join()
        return queue.get(), return_value

    def test_upload_input(self, random_byte_data: bytes):
        """
        Tests the `upload_input()` method.
        """
        assert (random_byte_data, True) == self._receive_request(
            ServerAction.UPLOADING_INPUT.to_bytes(),
            "upload_input",
            random_byte_data,
            MessageType.DATA,
        )

    @pytest.mark.parametrize(
        "method, action, message_type",
        [
            ("upload_model", ServerAction.UPLOADING_MODEL, MessageType.MODEL),
            (
                "upload_io_specification",
                ServerAction.UPLOADING_IOSPEC,
                MessageType.IO_SPEC,
            ),
        ],
    )
    def test_upload_with_path(
        self,
        tmpfolder: Path,
        random_byte_data: bytes,
        method: str,
        action: ServerAction,
        message_type: MessageType,
    ):
        """
        Tests the `upload_model()` method.
        """
        path = tmpfolder / uuid.uuid4().hex
        with open(path, "wb") as file:
            file.write(random_byte_data)

        assert (random_byte_data, True) == self._receive_request(
            action.to_bytes(),
            method,
            path,
            message_type,
        )

    def test_upload_runtime(self, tmpfolder: Path, random_byte_data: bytes):
        path = tmpfolder / uuid.uuid4().hex
        with open(path, "wb") as file:
            file.write(random_byte_data)

        assert (
            len(random_byte_data).to_bytes(4, "little") + random_byte_data,
            True,
        ) == self._receive_request(
            ServerAction.UPLOADING_RUNTIME.to_bytes(),
            "upload_runtime",
            path,
            MessageType.RUNTIME,
        )

    def test_download_output(
        self,
        random_byte_data: bytes,
    ):
        """
        Tests the `download_output()` method.
        """
        assert (b"", (True, random_byte_data)) == self._receive_request(
            random_byte_data,
            "download_output",
            None,
            MessageType.OUTPUT,
        )

    def test_download_statistics(self):
        """
        Tests the `download_statistics()` method.
        """
        data = {"1": "one", "2": "two", "3": "three"}
        to_send = json.dumps(data).encode()
        sent_bytes, downloaded_stats = self._receive_request(
            to_send,
            "download_statistics",
            True,
            MessageType.STATS,
        )
        assert b"" == sent_bytes
        assert isinstance(downloaded_stats, Measurements)
        assert downloaded_stats.data == data

    def test_disconnect(self, server_and_client):
        """
        Tests the `disconnect()` method.
        """
        server, client = server_and_client
        server.stop()
        client.stop()
        assert client.send_message(Message(MessageType.MODEL))
        assert server.send_message(Message(MessageType.MODEL))
        client.disconnect()
        with pytest.raises(ProtocolNotStartedError):
            client.send_message(Message(MessageType.MODEL))
        with pytest.raises(ConnectionResetError):
            server.send_message(Message(MessageType.MODEL))
        server.disconnect()
        with pytest.raises(ProtocolNotStartedError):
            server.send_message(Message(MessageType.MODEL))
