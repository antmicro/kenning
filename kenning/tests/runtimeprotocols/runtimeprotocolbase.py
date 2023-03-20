# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from kenning.core.measurements import Measurements
from kenning.core.runtimeprotocol import Message
from kenning.core.runtimeprotocol import MessageType
from kenning.core.runtimeprotocol import RuntimeProtocol, ServerStatus
from test_coreprotocol import TestCoreRuntimeProtocol
import json
import multiprocessing
import pytest
import socket
import time
import uuid


@pytest.mark.xdist_group(name='use_socket')
class RuntimeProtocolTests(TestCoreRuntimeProtocol):
    def test_initialize_server(self):
        """
        Tests the `initialize_server()` method.
        """
        server = self.initprotocol()
        assert server.initialize_server() is True
        second_server = self.initprotocol()
        assert second_server.initialize_server() is False
        assert second_server.serversocket is None
        server.disconnect()

    def test_initialize_client(self):
        """
        Tests the `initialize_client()` method.
        """
        client = self.initprotocol()
        with pytest.raises(ConnectionRefusedError):
            client.initialize_client()
        server = self.initprotocol()
        assert server.initialize_server() is True
        assert client.initialize_client() is True
        client.disconnect()
        server.disconnect()

    def test_receive_message(self, serverandclient):
        """
        Tests client status using `receive_message()` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient

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

    def test_receive_message_send_data(self, serverandclient):
        """
        Tests the `receive_message()` method by sending data.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient
        status, message = server.receive_message(timeout=1)
        assert status == ServerStatus.CLIENT_CONNECTED
        assert message is None

        # Send data
        data = self.generate_byte_data()
        client.send_message(Message(MessageType.OK, data))
        status, message = server.receive_message(timeout=1)
        assert status == ServerStatus.DATA_READY
        assert (message.payload == data and
                message.message_type == MessageType.OK)

    def test_receive_message_send_error(self, serverandclient):
        """
        Tests the `receive_message()` method by sending error message.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        data = self.generate_byte_data()
        server, client = serverandclient
        status, message = server.receive_message(timeout=1)
        assert status == ServerStatus.CLIENT_CONNECTED
        assert message is None

        # Send error message
        client.send_message(Message(MessageType.ERROR, data))
        status, message = server.receive_message(timeout=1)
        assert status == ServerStatus.DATA_READY
        assert (message.payload == data and
                message.message_type == MessageType.ERROR)

    def test_receive_message_send_empty(self, serverandclient):
        """
        Tests the `receive_message()` method by sending empty message.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient
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

    def test_send_data(self, serverandclient):
        """
        Tests the `send_data()` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient
        data = self.generate_byte_data()
        assert client.send_data(data) is True

    def test_receive_data(self, serverandclient):
        """
        Tests the `receive_data()` method with not initialized client.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient
        with pytest.raises(AttributeError):
            server.receive_data(None, None)

    def test_receive_data_data_sent(self, serverandclient):
        """
        Tests the `receive_data()` method with data being sent.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient
        data = self.generate_byte_data()
        server.accept_client(server.serversocket, None)

        assert client.send_data(data) is True
        status, received_data = server.receive_data(None, None)
        assert status is ServerStatus.DATA_READY
        assert data == received_data

    def test_receive_client_disconnect(self, serverandclient):
        """
        Tests the `receive_data()` method with client being disconnected.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient
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
            Tuple['ServerStatus', bytes] : Client addition status
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
        protocol = self.initprotocol()
        protocol.socket = True
        assert run_test(protocol)[0] == ServerStatus.CLIENT_IGNORED

        # There was no connection yet
        protocol = self.initprotocol()
        assert run_test(protocol)[0] == ServerStatus.CLIENT_CONNECTED

    def test_wait_send(self, serverandclient):
        """
        Tests the `wait_send()` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient
        server.accept_client(server.serversocket, None)
        data = self.generate_byte_data()

        for i in range(10):
            # Send the data
            client_out = client.wait_send(data)
            assert client_out == len(data)

            # Receive data
            server_status, server_data = server.receive_data(None, None)
            assert server_status == ServerStatus.DATA_READY
            assert server_data == data, f'{server_data}!={data}'

    def test_send_message(self, serverandclient):
        """
        Tests the `send_message(Message())` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient
        server.accept_client(server.serversocket, None)
        data = self.generate_byte_data()

        assert client.send_message(Message(MessageType.DATA, data)) is True
        assert server.send_message(Message(MessageType.DATA, data)) is True
        client.disconnect()
        with pytest.raises(ConnectionResetError):
            server.send_message(Message(MessageType.OK))

    @pytest.mark.parametrize('message,expected', [
        ((MessageType.OK, b''), (True, b'')),
        ((MessageType.ERROR, b''), (False, None)),
        ((MessageType.DATA, b''), (False, None)),
        ((MessageType.MODEL, b''), (False, None)),
        ((MessageType.PROCESS, b''), (False, None)),
        ((MessageType.OUTPUT, b''), (False, None)),
        ((MessageType.STATS, b''), (False, None)),
        ((MessageType.IOSPEC, b''), (False, None)),
        ])
    def test_receive_confirmation(self, serverandclient, message, expected):
        """
        Tests the `receive_confirmation()` method.

        Parameters
        ----------
        message : Tuple[MessageType, bytes]
            Message to be sent
        expected : Tuple[bool, Any]
            Expected output

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient
        client.send_message(Message(*message))
        output = server.receive_confirmation()
        assert output == expected

        # Check if client is disconnected
        client.disconnect()
        output = server.receive_confirmation()
        assert output == (False, None)

    def test_upload_input(self, serverandclient):
        """
        Tests the `upload_input()` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient

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
        data = self.generate_byte_data()
        shared_list = (multiprocessing.Manager()).list()
        thread = multiprocessing.Process(target=upload,
                                         args=(client, data, shared_list))

        thread.start()
        time.sleep(0.1)
        status, message = server.receive_message(timeout=1)
        server.send_message(Message(MessageType.OK, b''))
        thread.join()
        assert status == ServerStatus.DATA_READY
        assert message.payload == data
        assert shared_list[0] is True

    def test_upload_model(self, serverandclient, tmpfolder):
        """
        Tests the `upload_model()` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        tmpfolder : Path
            Fixture to get folder for model.
        """
        server, client = serverandclient
        path = tmpfolder / uuid.uuid4().hex
        data = self.generate_byte_data()
        with open(path, "wb") as file:
            file.write(data)

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
            status, received_model = server.receive_data(None, None)
            shared_list.append((status, received_model))
            output = server.send_message(Message(MessageType.OK))
            shared_list.append(output)

        shared_list = (multiprocessing.Manager()).list()
        thread_receive = multiprocessing.Process(target=receive_model,
                                                 args=(server, shared_list))
        server.accept_client(server.serversocket, None)
        thread_receive.start()
        assert client.upload_model(path) is True
        thread_receive.join()
        assert shared_list[1] is True
        receive_status, received_data = shared_list[0]
        assert receive_status == ServerStatus.DATA_READY
        answer = Message(MessageType.MODEL, data)
        parsed_message = client.parse_message(received_data)
        assert parsed_message == answer, f'{parsed_message}!={answer}'

    def test_upload_io_specification(self, serverandclient, tmpfolder):
        """
        Tests the `upload_io_specification()` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        tmpfolder : Path
            Fixture to get folder for io_specification.
        """
        # FIXME: Add actual example with input/output data
        server, client = serverandclient
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
            status, received = server.receive_data(None, None)
            shared_list.append((status, received))
            output = server.send_message(Message(MessageType.OK))
            shared_list.append(output)

        shared_list = (multiprocessing.Manager()).list()
        args = (server, shared_list)
        thread_receive = multiprocessing.Process(target=receive_io,
                                                 args=args)
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

    def test_download_output(self, serverandclient):
        """
        Tests the `download_output()` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient
        data = self.generate_byte_data()
        server.accept_client(server.serversocket, None)

        assert server.send_message(Message(MessageType.OK, data)) is True
        status, downloaded_data = client.download_output()
        assert status is True
        assert downloaded_data == data

    def test_download_statistics(self, serverandclient):
        """
        Tests the `download_statistics()` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient
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
        assert (message.message_type == MessageType.STATS and
                message.payload == b'')
        assert server.send_message(Message(MessageType.OK, to_send)) is True
        thread_send.join()

        downloaded_stats = shared_list[0]
        assert isinstance(downloaded_stats, Measurements)
        assert downloaded_stats.data == data

    @pytest.mark.parametrize('messagetype', [MessageType.OK, MessageType.ERROR,
                                             MessageType.DATA,
                                             MessageType.MODEL,
                                             MessageType.PROCESS,
                                             MessageType.STATS,
                                             MessageType.OUTPUT,
                                             MessageType.IOSPEC])
    def test_parse_message(self, serverandclient, messagetype):
        """
        Tests the `parse_message()` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        messagetype : MessageType
            A MessageType to send along with data
        """
        data = self.generate_byte_data()
        server, client = serverandclient
        server.accept_client(server.serversocket, None)

        client.send_message(Message(messagetype, data))
        status, message = server.receive_message(timeout=1)
        assert status == ServerStatus.DATA_READY
        assert data == message.payload and messagetype == message.message_type

    def test_disconnect(self, serverandclient):
        """
        Tests the `disconnect()` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient
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

    def test_request_processing(self, serverandclient):
        """
        Tests the `request_processing()` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient
        server.accept_client(server.serversocket, None)

        def send_ok(client):
            client.send_message(Message(MessageType.OK))

        def send_error(client):
            client.send_message(Message(MessageType.ERROR))

        functions = (send_ok, send_error)
        expected = (True, False)
        args = (client, )

        for function, expected in zip(functions, expected):
            thread_send = multiprocessing.Process(target=function, args=args)
            thread_send.start()
            response = server.request_processing()
            assert response is expected, f'{response}!={expected}'
            thread_send.join()

    def test_request_failure(self, serverandclient):
        """
        Tests the `request_failure()` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient
        server.accept_client(server.serversocket, None)
        server.request_failure()
        status, message = client.receive_data(None, None)
        assert status == ServerStatus.DATA_READY
        message = client.parse_message(message)
        assert (message.message_type == MessageType.ERROR and
                message.payload == b'')

    def test_request_success(self, serverandclient):
        """
        Tests the `request_success()` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient
        server.accept_client(server.serversocket, None)
        data = self.generate_byte_data()
        server.request_success(data)
        status, message = client.receive_data(None, None)
        assert status == ServerStatus.DATA_READY
        message = client.parse_message(message)
        assert (message.message_type == MessageType.OK and
                message.payload == data)
