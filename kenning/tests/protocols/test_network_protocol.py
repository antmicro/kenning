# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import json
import multiprocessing
import socket
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import pytest

from kenning.core.measurements import Measurements
from kenning.core.protocol import (
    Message,
    MessageType,
    Protocol,
    ServerStatus,
)
from kenning.protocols.network import NetworkProtocol
from kenning.tests.protocols.conftest import random_network_port
from kenning.tests.protocols.test_core_protocol import (
    TestCoreProtocol,
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
        self, server_and_client: Tuple[NetworkProtocol, NetworkProtocol]
    ):
        """
        Tests client status using `receive_message()` method.
        """
        server, client = server_and_client

        # Connect via Client
        status, message = server.receive_message(timeout=1)
        assert status == ServerStatus.CLIENT_CONNECTED
        assert message is None

        # Disconnect client
        client.disconnect()
        status, message = server.receive_message(timeout=1)
        assert status == ServerStatus.CLIENT_DISCONNECTED
        assert message is None

        # Timeout is reached
        status, message = server.receive_message(timeout=1)
        assert status == ServerStatus.NOTHING and message is None

    @pytest.mark.xdist_group(name="use_socket")
    def test_receive_message_send_data(
        self,
        server_and_client: Tuple[NetworkProtocol, NetworkProtocol],
        random_byte_data: bytes,
    ):
        """
        Tests the `receive_message()` method by sending data.
        """
        server, client = server_and_client
        status, message = server.receive_message(timeout=1)
        assert status == ServerStatus.CLIENT_CONNECTED
        assert message is None

        # Send data
        client.send_message(Message(MessageType.OK, random_byte_data))
        status, message = server.receive_message(timeout=1)
        assert status == ServerStatus.DATA_READY
        assert (
            message.payload == random_byte_data
            and message.message_type == MessageType.OK
        )

    @pytest.mark.xdist_group(name="use_socket")
    def test_receive_message_send_error(
        self,
        server_and_client: Tuple[NetworkProtocol, NetworkProtocol],
        random_byte_data: bytes,
    ):
        """
        Tests the `receive_message()` method by sending error message.
        """
        server, client = server_and_client
        status, message = server.receive_message(timeout=1)
        assert status == ServerStatus.CLIENT_CONNECTED
        assert message is None

        # Send error message
        client.send_message(Message(MessageType.ERROR, random_byte_data))
        status, message = server.receive_message(timeout=1)
        assert status == ServerStatus.DATA_READY
        assert (
            message.payload == random_byte_data
            and message.message_type == MessageType.ERROR
        )

    @pytest.mark.xdist_group(name="use_socket")
    def test_receive_message_send_empty(
        self, server_and_client: Tuple[NetworkProtocol, NetworkProtocol]
    ):
        """
        Tests the `receive_message()` method by sending empty message.
        """
        server, client = server_and_client
        status, message = server.receive_message(timeout=1)
        assert status == ServerStatus.CLIENT_CONNECTED
        assert message is None

        # Send empty message
        class EmptyMessage(object):
            def to_bytes(self):
                return b""

        client.send_message(EmptyMessage())
        status, message = server.receive_message(timeout=1)
        assert message is None
        assert status == ServerStatus.NOTHING, status

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
        assert client.send_data(random_byte_data)

    @pytest.mark.xdist_group(name="use_socket")
    def test_receive_data(
        self, server_and_client: Tuple[NetworkProtocol, NetworkProtocol]
    ):
        """
        Tests the `receive_data()` method with not initialized client.
        """
        server, client = server_and_client
        with pytest.raises(AttributeError):
            server.receive_data(None, None)

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
        server.accept_client(server.serversocket, None)

        assert client.send_data(random_byte_data)
        status, received_data = server.receive_data(None, None)
        assert status is ServerStatus.DATA_READY
        assert random_byte_data == received_data

    @pytest.mark.xdist_group(name="use_socket")
    def test_receive_client_disconnect(
        self, server_and_client: Tuple[NetworkProtocol, NetworkProtocol]
    ):
        """
        Tests the `receive_data()` method with client being disconnected.
        """
        server, client = server_and_client
        server.accept_client(server.serversocket, None)

        client.disconnect()
        status, received_data = server.receive_data(None, None)
        assert status == ServerStatus.CLIENT_DISCONNECTED
        assert received_data is None

    @pytest.mark.xdist_group(name="use_socket")
    def test_accept_client(self):
        """
        Tests the `accept_client()` method.
        """

        def connect():
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
            Tuple['ServerStatus', bytes] :
                Client addition status
            """
            output: ServerStatus
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.host, self.port))
                s.listen(1)
                multiprocessing.Process(target=connect).start()
                output = protocol.accept_client(s, None)
                s.shutdown(socket.SHUT_RDWR)
            return output

        # There's already established connection
        protocol = self.init_protocol()
        protocol.socket = True
        assert run_test(protocol)[0] == ServerStatus.CLIENT_IGNORED

        # There was no connection yet
        protocol = self.init_protocol()
        assert run_test(protocol)[0] == ServerStatus.CLIENT_CONNECTED

    @pytest.mark.xdist_group(name="use_socket")
    def test_wait_send(
        self,
        server_and_client: Tuple[NetworkProtocol, NetworkProtocol],
        random_byte_data: bytes,
    ):
        """
        Tests the `wait_send()` method.
        """
        server, client = server_and_client
        server.accept_client(server.serversocket, None)

        for _ in range(10):
            # Send the data
            client_out = client.wait_send(random_byte_data)
            assert client_out == len(random_byte_data)

            # Receive data
            server_status, server_data = server.receive_data(None, None)
            assert server_status == ServerStatus.DATA_READY
            assert (
                server_data == random_byte_data
            ), f"{server_data}!={random_byte_data}"

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
        server.accept_client(server.serversocket, None)

        assert client.send_message(Message(MessageType.DATA, random_byte_data))
        assert server.send_message(Message(MessageType.DATA, random_byte_data))

        client.disconnect()
        with pytest.raises(ConnectionResetError):
            server.send_message(Message(MessageType.OK))

    @pytest.mark.xdist_group(name="use_socket")
    @pytest.mark.parametrize(
        "message,expected",
        [
            (Message(MessageType.OK), (True, b"")),
            (Message(MessageType.ERROR), (False, None)),
            (Message(MessageType.DATA), (False, None)),
            (Message(MessageType.MODEL), (False, None)),
            (Message(MessageType.PROCESS), (False, None)),
            (Message(MessageType.OUTPUT), (False, None)),
            (Message(MessageType.STATS), (False, None)),
            (Message(MessageType.IO_SPEC), (False, None)),
        ],
    )
    def test_receive_confirmation(
        self,
        server_and_client: Tuple[NetworkProtocol, NetworkProtocol],
        message: Message,
        expected: Tuple[bool, Optional[bytes]],
    ):
        """
        Tests the `receive_confirmation()` method.
        """
        server, client = server_and_client
        client.send_message(message)
        output = server.receive_confirmation()
        assert output == expected

        # Check if client is disconnected
        client.disconnect()
        output = server.receive_confirmation()
        assert output == (False, None)

    @pytest.mark.xdist_group(name="use_socket")
    def test_upload_input(
        self,
        server_and_client: Tuple[NetworkProtocol, NetworkProtocol],
        random_byte_data: bytes,
    ):
        """
        Tests the `upload_input()` method.
        """
        server, client = server_and_client

        def upload(
            client: NetworkProtocol,
            data: bytes,
            queue: multiprocessing.Queue,
        ):
            """
            Waits for confirmation message and sends input data.

            Parameters
            ----------
            client : NetworkProtocol
                Initialized NetworkProtocol client
            data : bytes
                Input data to be sent
            queue : multiprocessing.Queue
                Shared list to append output of `upload_input()` method
            """
            output = client.upload_input(data)
            queue.put(output)

        server.accept_client(server.serversocket, None)
        queue = multiprocessing.Queue()
        thread = multiprocessing.Process(
            target=upload, args=(client, random_byte_data, queue)
        )

        thread.start()
        status, message = server.receive_message(timeout=1)
        server.send_message(Message(MessageType.OK))
        thread.join()
        assert status == ServerStatus.DATA_READY
        assert message.payload == random_byte_data
        assert queue.get()

    @pytest.mark.xdist_group(name="use_socket")
    def test_upload_model(
        self,
        server_and_client: Tuple[NetworkProtocol, NetworkProtocol],
        tmpfolder: Path,
        random_byte_data: bytes,
    ):
        """
        Tests the `upload_model()` method.
        """
        server, client = server_and_client
        path = tmpfolder / uuid.uuid4().hex
        with open(path, "wb") as file:
            file.write(random_byte_data)

        def receive_model(
            server: NetworkProtocol, queue: multiprocessing.Queue
        ):
            """
            Receives uploaded model.

            Parameters
            ----------
            server : NetworkProtocol
                Initialized NetworkProtocol server.
            queue : multiprocessing.Queue
                Shared list to to append received data.
            """
            time.sleep(0.1)
            status, received_model = server.receive_data(None, None)
            queue.put((status, received_model))
            output = server.send_message(Message(MessageType.OK))
            queue.put(output)

        queue = multiprocessing.Queue()
        thread_receive = multiprocessing.Process(
            target=receive_model, args=(server, queue)
        )
        server.accept_client(server.serversocket, None)
        thread_receive.start()
        assert client.upload_model(path)
        thread_receive.join()
        receive_status, received_data = queue.get()
        assert queue.get()
        assert receive_status == ServerStatus.DATA_READY
        answer = Message(MessageType.MODEL, random_byte_data)
        parsed_message = client.parse_message(received_data)
        assert parsed_message == answer, f"{parsed_message}!={answer}"

    @pytest.mark.xdist_group(name="use_socket")
    def test_upload_io_specification(
        self,
        server_and_client: Tuple[NetworkProtocol, NetworkProtocol],
        tmpfolder: Path,
    ):
        """
        Tests the `upload_io_specification()` method.
        """
        # FIXME: Add actual example with input/output data
        server, client = server_and_client
        io_specification = {1: "one", 2: "two", 3: "three"}
        path = tmpfolder / uuid.uuid4().hex
        with open(path, "w") as file:
            json.dump(io_specification, file)

        def receive_io(server: NetworkProtocol, queue: multiprocessing.Queue):
            """
            Receives input/output details.

            Parameters
            ----------
            server : NetworkProtocol
                Initialized NetworkProtocol server
            queue : multiprocessing.Queue
                Shared list to append received input/output details
            """
            time.sleep(0.1)
            status, received = server.receive_data(None, None)
            queue.put((status, received))
            output = server.send_message(Message(MessageType.OK))
            queue.put(output)

        queue = multiprocessing.Queue()
        thread_receive = multiprocessing.Process(
            target=receive_io, args=(server, queue)
        )
        server.accept_client(server.serversocket, None)

        thread_receive.start()
        assert client.upload_io_specification(path)
        thread_receive.join()

        receive_status, received_data = queue.get()
        send_message_status = queue.get()
        assert send_message_status
        assert receive_status == ServerStatus.DATA_READY
        encoded_data = (json.dumps(io_specification)).encode()
        answer = Message(MessageType.IO_SPEC, encoded_data)
        parsed_message = client.parse_message(received_data)
        assert parsed_message == answer, f"{parsed_message}!={answer}"

    @pytest.mark.xdist_group(name="use_socket")
    def test_download_output(
        self,
        server_and_client: Tuple[NetworkProtocol, NetworkProtocol],
        random_byte_data: bytes,
    ):
        """
        Tests the `download_output()` method.
        """
        server, client = server_and_client
        server.accept_client(server.serversocket, None)

        assert server.send_message(Message(MessageType.OK, random_byte_data))
        status, downloaded_data = client.download_output()
        assert status
        assert downloaded_data == random_byte_data

    @pytest.mark.xdist_group(name="use_socket")
    def test_download_statistics(
        self, server_and_client: Tuple[NetworkProtocol, NetworkProtocol]
    ):
        """
        Tests the `download_statistics()` method.
        """
        server, client = server_and_client
        data = {"1": "one", "2": "two", "3": "three"}
        to_send = json.dumps(data).encode()
        server.accept_client(server.serversocket, None)

        def download_stats(
            client: NetworkProtocol, queue: multiprocessing.Queue
        ):
            """
            Downloads statistics sent by server.

            Parameters
            ----------
            client : NetworkProtocol
                Initialized NetworkProtocol client
            queue : multiprocessing.Queue
                Shared list to append downloaded statistics
            """
            time.sleep(0.1)
            client.send_message(Message(MessageType.OK))
            client.receive_confirmation()
            output = client.download_statistics()
            queue.put(output)

        queue = multiprocessing.Queue()
        thread_send = multiprocessing.Process(
            target=download_stats, args=(client, queue)
        )
        thread_send.start()

        output = server.receive_confirmation()
        assert output == (True, b"")
        server.send_message(Message(MessageType.OK))

        time.sleep(0.1)
        status, message = server.receive_message(timeout=1)
        assert status == ServerStatus.DATA_READY
        assert (
            message.message_type == MessageType.STATS
            and message.payload == b""
        )
        assert server.send_message(Message(MessageType.OK, to_send))
        thread_send.join()

        downloaded_stats = queue.get()
        assert isinstance(downloaded_stats, Measurements)
        assert downloaded_stats.data == data

    @pytest.mark.xdist_group(name="use_socket")
    @pytest.mark.parametrize(
        "message_type",
        [
            MessageType.OK,
            MessageType.ERROR,
            MessageType.DATA,
            MessageType.MODEL,
            MessageType.PROCESS,
            MessageType.STATS,
            MessageType.OUTPUT,
            MessageType.IO_SPEC,
        ],
    )
    def test_parse_message(
        self,
        server_and_client: Tuple[NetworkProtocol, NetworkProtocol],
        message_type: MessageType,
        random_byte_data: bytes,
    ):
        """
        Tests the `parse_message()` method.
        """
        server, client = server_and_client
        server.accept_client(server.serversocket, None)

        client.send_message(Message(message_type, random_byte_data))
        status, message = server.receive_message(timeout=1)
        assert status == ServerStatus.DATA_READY
        assert (
            random_byte_data == message.payload
            and message_type == message.message_type
        )

    @pytest.mark.xdist_group(name="use_socket")
    def test_disconnect(self, server_and_client):
        """
        Tests the `disconnect()` method.
        """
        server, client = server_and_client
        server.accept_client(server.serversocket, None)
        assert client.send_message(Message(MessageType.OK))
        assert server.send_message(Message(MessageType.OK))
        client.disconnect()
        with pytest.raises(OSError):
            client.send_message(Message(MessageType.OK))
        with pytest.raises(ConnectionResetError):
            server.send_message(Message(MessageType.OK))
        server.disconnect()
        with pytest.raises(OSError):
            server.send_message(Message(MessageType.OK))

    @pytest.mark.xdist_group(name="use_socket")
    @pytest.mark.parametrize(
        "client_response,expected",
        [(Message(MessageType.OK), True), (Message(MessageType.ERROR), False)],
    )
    def test_request_processing(
        self,
        server_and_client: Tuple[NetworkProtocol, NetworkProtocol],
        client_response: Message,
        expected: bool,
    ):
        """
        Tests the `request_processing()` method.
        """
        server, client = server_and_client
        server.accept_client(server.serversocket, None)

        thread_send = multiprocessing.Process(
            target=client.send_message, args=(client_response,)
        )
        thread_send.start()
        response = server.request_processing()
        assert response is expected, f"{response}!={expected}"
        thread_send.join()

    @pytest.mark.xdist_group(name="use_socket")
    def test_request_failure(self, server_and_client):
        """
        Tests the `request_failure()` method.
        """
        server, client = server_and_client
        server.accept_client(server.serversocket, None)
        server.request_failure()
        status, message = client.receive_data(None, None)
        assert status == ServerStatus.DATA_READY
        message = client.parse_message(message)
        assert (
            message.message_type == MessageType.ERROR
            and message.payload == b""
        )

    @pytest.mark.xdist_group(name="use_socket")
    def test_request_success(
        self,
        server_and_client: Tuple[NetworkProtocol, NetworkProtocol],
        random_byte_data: bytes,
    ):
        """
        Tests the `request_success()` method.
        """
        server, client = server_and_client
        server.accept_client(server.serversocket, None)
        server.request_success(random_byte_data)
        status, message = client.receive_data(None, None)
        assert status == ServerStatus.DATA_READY
        message = client.parse_message(message)
        assert (
            message.message_type == MessageType.OK
            and message.payload == random_byte_data
        )
