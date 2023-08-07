# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import json
import multiprocessing
import socket
import time
import uuid

import pytest

from kenning.core.measurements import Measurements
from kenning.core.runtimeprotocol import (
    Message,
    MessageType,
    RuntimeProtocol,
    ServerStatus,
)
from kenning.runtimeprotocols.network import NetworkProtocol
from kenning.tests.runtimeprotocols.conftest import random_network_port
from kenning.tests.runtimeprotocols.test_core_protocol import (
    TestCoreRuntimeProtocol,
)


class TestNetworkProtocol(TestCoreRuntimeProtocol):
    host = 'localhost'
    port = random_network_port()

    def init_protocol(self):
        return NetworkProtocol(self.host, self.port)

    def test_initialize_server(self):
        """
        Tests the `initialize_server()` method.
        """
        server = self.init_protocol()
        assert server.initialize_server() is True
        second_server = self.init_protocol()
        assert second_server.initialize_server() is False
        assert second_server.serversocket is None
        server.disconnect()

    def test_initialize_client(self):
        """
        Tests the `initialize_client()` method.
        """
        client = self.init_protocol()
        with pytest.raises(ConnectionRefusedError):
            client.initialize_client()
        server = self.init_protocol()
        assert server.initialize_server() is True
        assert client.initialize_client() is True
        client.disconnect()
        server.disconnect()

    def test_receive_message(self, server_and_client):
        """
        Tests client status using `receive_message()` method.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
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

    def test_receive_message_send_data(
        self, server_and_client, random_byte_data
    ):
        """
        Tests the `receive_message()` method by sending data.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
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

    def test_receive_message_send_error(
        self, server_and_client, random_byte_data
    ):
        """
        Tests the `receive_message()` method by sending error message.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
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

    def test_receive_message_send_empty(self, server_and_client):
        """
        Tests the `receive_message()` method by sending empty message.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = server_and_client
        status, message = server.receive_message(timeout=1)
        assert status == ServerStatus.CLIENT_CONNECTED
        assert message is None

        # Send empty message
        class EmptyMessage(object):
            def to_bytes(self):
                return b''

        client.send_message(EmptyMessage())
        status, message = server.receive_message(timeout=1)
        assert message is None
        assert status == ServerStatus.NOTHING, status

    def test_send_data(self, server_and_client, random_byte_data):
        """
        Tests the `send_data()` method.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = server_and_client
        assert client.send_data(random_byte_data) is True

    def test_receive_data(self, server_and_client):
        """
        Tests the `receive_data()` method with not initialized client.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = server_and_client
        with pytest.raises(AttributeError):
            server.receive_data(None, None)

    def test_receive_data_data_sent(self, server_and_client, random_byte_data):
        """
        Tests the `receive_data()` method with data being sent.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = server_and_client
        server.accept_client(server.serversocket, None)

        assert client.send_data(random_byte_data) is True
        status, received_data = server.receive_data(None, None)
        assert status is ServerStatus.DATA_READY
        assert random_byte_data == received_data

    def test_receive_client_disconnect(self, server_and_client):
        """
        Tests the `receive_data()` method with client being disconnected.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = server_and_client
        server.accept_client(server.serversocket, None)

        client.disconnect()
        status, received_data = server.receive_data(None, None)
        assert status == ServerStatus.CLIENT_DISCONNECTED
        assert received_data is None

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

        def run_test(protocol: RuntimeProtocol):
            """
            Initializes socket and conncets to it.

            Parameters
            ----------
            protocol : RuntimeProtocol
                Initialized RuntimeProtocol object

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

    def test_wait_send(self, server_and_client, random_byte_data):
        """
        Tests the `wait_send()` method.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = server_and_client
        server.accept_client(server.serversocket, None)

        for i in range(10):
            # Send the data
            client_out = client.wait_send(random_byte_data)
            assert client_out == len(random_byte_data)

            # Receive data
            server_status, server_data = server.receive_data(None, None)
            assert server_status == ServerStatus.DATA_READY
            assert (
                server_data == random_byte_data
            ), f'{server_data}!={random_byte_data}'

    def test_send_message(self, server_and_client, random_byte_data):
        """
        Tests the `send_message(Message())` method.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = server_and_client
        server.accept_client(server.serversocket, None)

        assert (
            client.send_message(Message(MessageType.DATA, random_byte_data))
            is True
        )
        assert (
            server.send_message(Message(MessageType.DATA, random_byte_data))
            is True
        )
        client.disconnect()
        with pytest.raises(ConnectionResetError):
            server.send_message(Message(MessageType.OK))

    @pytest.mark.parametrize(
        'message,expected',
        [
            ((MessageType.OK, b''), (True, b'')),
            ((MessageType.ERROR, b''), (False, None)),
            ((MessageType.DATA, b''), (False, None)),
            ((MessageType.MODEL, b''), (False, None)),
            ((MessageType.PROCESS, b''), (False, None)),
            ((MessageType.OUTPUT, b''), (False, None)),
            ((MessageType.STATS, b''), (False, None)),
            ((MessageType.IOSPEC, b''), (False, None)),
        ],
    )
    def test_receive_confirmation(self, server_and_client, message, expected):
        """
        Tests the `receive_confirmation()` method.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        message : Tuple[MessageType, bytes]
            Message to be sent
        expected : Tuple[bool, Any]
            Expected output
        """
        server, client = server_and_client
        client.send_message(Message(*message))
        output = server.receive_confirmation()
        assert output == expected

        # Check if client is disconnected
        client.disconnect()
        output = server.receive_confirmation()
        assert output == (False, None)

    def test_upload_input(self, server_and_client, random_byte_data):
        """
        Tests the `upload_input()` method.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = server_and_client

        def upload(client, data, shared_list):
            """
            Waits for confirmation message and sends input data.

            Parameters
            ----------
            client : RuntimeProtocol
                Initialized RuntimeProtocol client
            data : bytes
                Input data to be sent
            shared_list : List
                Shared list to append output of `upload_input()` method
            """
            output = client.upload_input(data)
            shared_list.append(output)

        server.accept_client(server.serversocket, None)
        shared_list = (multiprocessing.Manager()).list()
        thread = multiprocessing.Process(
            target=upload, args=(client, random_byte_data, shared_list)
        )

        thread.start()
        status, message = server.receive_message(timeout=1)
        server.send_message(Message(MessageType.OK, b''))
        thread.join()
        assert status == ServerStatus.DATA_READY
        assert message.payload == random_byte_data
        assert shared_list[0] is True

    def test_upload_model(
        self, server_and_client, tmpfolder, random_byte_data
    ):
        """
        Tests the `upload_model()` method.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        tmpfolder : Path
            Fixture to get folder for model.
        """
        server, client = server_and_client
        path = tmpfolder / uuid.uuid4().hex
        with open(path, "wb") as file:
            file.write(random_byte_data)

        def receive_model(server: RuntimeProtocol, shared_list: list):
            """
            Receives uploaded model.

            Parameters
            ----------
            server : RuntimeProtocol
                Initialized RuntimeProtocol server.
            shared_list : List
                Shared list to to append received data.
            """
            time.sleep(0.1)
            status, received_model = server.receive_data(None, None)
            shared_list.append((status, received_model))
            output = server.send_message(Message(MessageType.OK))
            shared_list.append(output)

        shared_list = (multiprocessing.Manager()).list()
        thread_receive = multiprocessing.Process(
            target=receive_model, args=(server, shared_list)
        )
        server.accept_client(server.serversocket, None)
        thread_receive.start()
        assert client.upload_model(path) is True
        thread_receive.join()
        assert shared_list[1] is True
        receive_status, received_data = shared_list[0]
        assert receive_status == ServerStatus.DATA_READY
        answer = Message(MessageType.MODEL, random_byte_data)
        parsed_message = client.parse_message(received_data)
        assert parsed_message == answer, f'{parsed_message}!={answer}'

    def test_upload_io_specification(self, server_and_client, tmpfolder):
        """
        Tests the `upload_io_specification()` method.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        tmpfolder : Path
            Fixture to get folder for io_specification.
        """
        # FIXME: Add actual example with input/output data
        server, client = server_and_client
        io_specification = {1: 'one', 2: 'two', 3: 'three'}
        path = tmpfolder / uuid.uuid4().hex
        with open(path, 'w') as file:
            json.dump(io_specification, file)

        def receive_io(server: RuntimeProtocol, shared_list: list):
            """
            Receives input/output details.

            Parameters
            ----------
            server : RuntimeProtocol
                Initialized RuntimeProtocol server
            shared_list : List
                Shared list to append received input/output details
            """
            time.sleep(0.1)
            status, received = server.receive_data(None, None)
            shared_list.append((status, received))
            output = server.send_message(Message(MessageType.OK))
            shared_list.append(output)

        shared_list = (multiprocessing.Manager()).list()
        args = (server, shared_list)
        thread_receive = multiprocessing.Process(target=receive_io, args=args)
        server.accept_client(server.serversocket, None)

        thread_receive.start()
        assert client.upload_io_specification(path) is True
        thread_receive.join()

        receive_status, received_data = shared_list[0]
        send_message_status = shared_list[1]
        assert send_message_status is True
        assert receive_status == ServerStatus.DATA_READY
        encoded_data = (json.dumps(io_specification)).encode()
        answer = Message(MessageType.IOSPEC, encoded_data)
        parsed_message = client.parse_message(received_data)
        assert parsed_message == answer, f'{parsed_message}!={answer}'

    def test_download_output(self, server_and_client, random_byte_data):
        """
        Tests the `download_output()` method.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = server_and_client
        server.accept_client(server.serversocket, None)

        assert (
            server.send_message(Message(MessageType.OK, random_byte_data))
            is True
        )
        status, downloaded_data = client.download_output()
        assert status is True
        assert downloaded_data == random_byte_data

    def test_download_statistics(self, server_and_client):
        """
        Tests the `download_statistics()` method.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = server_and_client
        data = {'1': 'one', '2': 'two', '3': 'three'}
        to_send = json.dumps(data).encode()
        server.accept_client(server.serversocket, None)

        def download_stats(client, shared_list):
            """
            Downloads statistics sent by server.

            Parameters
            ----------
            client : RuntimeProtocol
                Initialized RuntimeProtocol client
            shared_list : List
                Shared list to append downloaded statistics
            """
            time.sleep(0.1)
            client.send_message(Message(MessageType.OK))
            client.receive_confirmation()
            output = client.download_statistics()
            shared_list.append(output)

        shared_list = (multiprocessing.Manager()).list()
        args = (client, shared_list)
        thread_send = multiprocessing.Process(target=download_stats, args=args)
        thread_send.start()

        output = server.receive_confirmation()
        assert output == (True, b'')
        server.send_message(Message(MessageType.OK))

        time.sleep(0.1)
        status, message = server.receive_message(timeout=1)
        assert status == ServerStatus.DATA_READY
        assert (
            message.message_type == MessageType.STATS
            and message.payload == b''
        )
        assert server.send_message(Message(MessageType.OK, to_send)) is True
        thread_send.join()

        downloaded_stats = shared_list[0]
        assert isinstance(downloaded_stats, Measurements)
        assert downloaded_stats.data == data

    @pytest.mark.parametrize(
        'message_type',
        [
            MessageType.OK,
            MessageType.ERROR,
            MessageType.DATA,
            MessageType.MODEL,
            MessageType.PROCESS,
            MessageType.STATS,
            MessageType.OUTPUT,
            MessageType.IOSPEC,
        ],
    )
    def test_parse_message(
        self, server_and_client, message_type, random_byte_data
    ):
        """
        Tests the `parse_message()` method.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        message_type : MessageType
            A MessageType to send along with data
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

    def test_disconnect(self, server_and_client):
        """
        Tests the `disconnect()` method.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = server_and_client
        server.accept_client(server.serversocket, None)
        assert client.send_message(Message(MessageType.OK)) is True
        assert server.send_message(Message(MessageType.OK)) is True
        client.disconnect()
        with pytest.raises(OSError):
            client.send_message(Message(MessageType.OK))
        with pytest.raises(ConnectionResetError):
            server.send_message(Message(MessageType.OK))
        server.disconnect()
        with pytest.raises(OSError):
            server.send_message(Message(MessageType.OK))

    def test_request_processing(self, server_and_client):
        """
        Tests the `request_processing()` method.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = server_and_client
        server.accept_client(server.serversocket, None)

        def send_ok(client):
            client.send_message(Message(MessageType.OK))

        def send_error(client):
            client.send_message(Message(MessageType.ERROR))

        functions = (send_ok, send_error)
        expected = (True, False)
        args = (client,)

        for function, expected in zip(functions, expected):
            thread_send = multiprocessing.Process(target=function, args=args)
            thread_send.start()
            response = server.request_processing()
            assert response is expected, f'{response}!={expected}'
            thread_send.join()

    def test_request_failure(self, server_and_client):
        """
        Tests the `request_failure()` method.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = server_and_client
        server.accept_client(server.serversocket, None)
        server.request_failure()
        status, message = client.receive_data(None, None)
        assert status == ServerStatus.DATA_READY
        message = client.parse_message(message)
        assert (
            message.message_type == MessageType.ERROR
            and message.payload == b''
        )

    def test_request_success(self, server_and_client, random_byte_data):
        """
        Tests the `request_success()` method.

        Parameters
        ----------
        server_and_client : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
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
