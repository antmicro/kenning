# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import json
import multiprocessing
import socket
import time
import uuid
from pathlib import Path
from typing import Any, Tuple

import pytest

from kenning.core.measurements import Measurements
from kenning.core.protocol import Protocol, ServerAction
from kenning.protocols.bytes_based_protocol import TransmissionFlag
from kenning.protocols.kenning_protocol import ProtocolNotStartedError
from kenning.protocols.message import (
    FlagName,
    Flags,
    FlowControlFlags,
    Message,
    MessageType,
)
from kenning.protocols.network import NetworkProtocol
from kenning.tests.protocols.conftest import random_network_port
from kenning.tests.protocols.test_core_protocol import (
    TestCoreProtocol,
)


def valid_status_message(action: ServerAction):
    return Message(
        MessageType.STATUS,
        action.to_bytes(),
        FlowControlFlags.TRANSMISSION,
        Flags(
            {
                FlagName.FIRST: True,
                FlagName.LAST: True,
                FlagName.SPEC_FLAG_2: True,
                FlagName.SUCCESS: True,
            }
        ),
    )


class TestNetworkProtocol(TestCoreProtocol):
    host = "localhost"
    port = random_network_port()

    def init_protocol(self):
        if self.port is None:
            pytest.fail("Cannot find free port")
        return NetworkProtocol(self.host, self.port)

    @pytest.mark.xdist_group(name="use_socket")
    def test_initialize_server(self):
        """
        Tests the `initialize_server()` method.
        """
        server = self.init_protocol()
        assert server.initialize_server()
        second_server = self.init_protocol()
        assert not second_server.initialize_server()
        assert second_server.serversocket is None
        server.disconnect()

    @pytest.mark.xdist_group(name="use_socket")
    def test_initialize_client(self):
        """
        Tests the `initialize_client()` method.
        """
        client = self.init_protocol()
        with pytest.raises(ConnectionRefusedError):
            client.initialize_client()
        server = self.init_protocol()
        assert server.initialize_server()
        assert client.initialize_client()
        client.disconnect()
        server.disconnect()

    @pytest.mark.xdist_group(name="use_socket")
    def test_receive_message(
        self,
        server_and_client: Tuple[NetworkProtocol, NetworkProtocol],
        random_byte_data: bytes,
    ):
        """
        Tests the `receive_message()` method by sending data.
        """
        server, client = server_and_client
        server.stop()
        message = server.receive_message(timeout=1)
        assert message is None

        # Send data
        client.send_message(Message(MessageType.OUTPUT, random_byte_data))
        message = server.receive_message(timeout=1)
        assert (
            message.payload == random_byte_data
            and message.message_type == MessageType.OUTPUT
        )

    @pytest.mark.xdist_group(name="use_socket")
    def test_receive_message_send_empty(
        self, server_and_client: Tuple[NetworkProtocol, NetworkProtocol]
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

    @pytest.mark.xdist_group(name="use_socket")
    def test_send_data(
        self,
        server_and_client: Tuple[NetworkProtocol, NetworkProtocol],
        random_byte_data: bytes,
    ):
        """
        Tests the `send_data()` method.
        """
        server, client = server_and_client
        server.stop()
        assert client.send_data(random_byte_data)

    @pytest.mark.xdist_group(name="use_socket")
    def test_receive_data(
        self, server_and_client: Tuple[NetworkProtocol, NetworkProtocol]
    ):
        """
        Tests the `receive_data()` method with not initialized server.
        """
        server, client = server_and_client
        server.stop()
        server.disconnect()
        with pytest.raises(ProtocolNotStartedError):
            server.receive_data(None)

    @pytest.mark.xdist_group(name="use_socket")
    def test_receive_data_data_sent(
        self,
        server_and_client: Tuple[NetworkProtocol, NetworkProtocol],
        random_byte_data: bytes,
    ):
        """
        Tests the `receive_data()` method with data being sent.
        """
        server, client = server_and_client
        server.stop()
        assert client.send_data(random_byte_data)
        received_data = server.receive_data(None)
        assert random_byte_data == received_data

    @pytest.mark.xdist_group(name="use_socket")
    def test_receive_client_disconnect(
        self, server_and_client: Tuple[NetworkProtocol, NetworkProtocol]
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

    @pytest.mark.xdist_group(name="use_socket")
    def test_accept_client(self):
        """
        Tests the `accept_client()` method.
        """

        def connect(s):
            """
            Connects to server-socket.
            """
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.host, self.port))
            s.close()

        def run_test(protocol: Protocol):
            """
            Initializes socket and conncets to it.

            Parameters
            ----------
            protocol : Protocol
                Initialized Protocol object

            Returns
            -------
            Tuple['ServerStatus', bytes]
                Client addition status
            """
            output = False
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.host, self.port))
                protocol.serversocket = s
                multiprocessing.Process(target=connect, args=(s,)).start()
                output = protocol.accept_client(None)
                s.shutdown(socket.SHUT_RDWR)
            return output

        # There's already established connection
        protocol = self.init_protocol()
        protocol.socket = True
        run_test(protocol)
        assert socket

        # There was no connection yet
        protocol = self.init_protocol()
        run_test(protocol)
        assert socket is not None

    @pytest.mark.xdist_group(name="use_socket")
    def test_send_message(
        self,
        server_and_client: Tuple[NetworkProtocol, NetworkProtocol],
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
            host: str,
            port: int,
            response_payload: bytes,
            server_started_event: multiprocessing.Event,
            queue: multiprocessing.Queue,
        ):
            server = NetworkProtocol(host, port)
            server.initialize_server()
            server_started_event.set()
            type, message_type, data, flags = server.listen_blocking(
                None, None, None, 1
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
                self.host,
                self.port,
                response_payload,
                server_started_event,
                queue,
            ),
        )
        thread.start()
        server_started_event.wait()
        client = NetworkProtocol(self.host, self.port)
        assert client.initialize_client()
        if argument is not None:
            return_value = getattr(client, method)(argument)
        else:
            return_value = getattr(client, method)()
        client.disconnect()
        thread.join()
        return queue.get(), return_value

    @pytest.mark.xdist_group(name="use_socket")
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
    @pytest.mark.xdist_group(name="use_socket")
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

    def test_request_processing(self):
        assert (b"", True) == self._receive_request(
            ServerAction.PROCESSING_INPUT.to_bytes(),
            "request_processing",
            time.perf_counter,
            MessageType.PROCESS,
        )

    @pytest.mark.xdist_group(name="use_socket")
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

    @pytest.mark.xdist_group(name="use_socket")
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

    @pytest.mark.xdist_group(name="use_socket")
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
