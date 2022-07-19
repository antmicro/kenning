from kenning.core.measurements import Measurements
from kenning.core.runtimeprotocol import MessageType
from kenning.core.runtimeprotocol import RuntimeProtocol, ServerStatus
from test_coreprotocol import TestCoreRuntimeProtocol
import json
import multiprocessing
import pytest
import socket
import uuid


class RuntimeProtocolTests(TestCoreRuntimeProtocol):
    def test_initialize_server(self):
        """
        Tests the `initialize_server()` method.
        """
        server = self.initprotocol()
        assert server.initialize_server() is True
        with pytest.raises(OSError) as execinfo:
            second_server = self.initprotocol()
            second_server.initialize_server()
        assert 'Address already in use' in str(execinfo.value)
        server.disconnect()

    def test_initialize_client(self):
        """
        Tests the `initialize_server()` method.
        """
        client = self.initprotocol()
        with pytest.raises(ConnectionRefusedError):
            client.initialize_client()
        server = self.initprotocol()
        assert server.initialize_server() is True
        assert client.initialize_client() is True
        client.disconnect()
        server.disconnect()

    def test_wait_for_activity(self, serverandclient):
        """
        Tests the `wait_for_activity()` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient

        # Connect via Client
        status, received_data = server.wait_for_activity()[0]
        assert status == ServerStatus.CLIENT_CONNECTED
        assert received_data is None

        # Send data
        data, _ = self.generate_byte_data()
        client.send_data(data)
        status, received_data = server.wait_for_activity()[0]
        assert status == ServerStatus.DATA_READY
        assert received_data == [data]

        # Send error message
        client.send_message(MessageType.ERROR, b'')
        status, received_data = server.wait_for_activity()[0]
        assert status == ServerStatus.DATA_READY
        assert received_data == [MessageType.ERROR.to_bytes()]

        # Send empty message
        class EmptyMessage:
            def to_bytes(self):
                return b''

        client.send_message(EmptyMessage(), b'')
        status, received_data = server.wait_for_activity()[0]
        assert received_data == [b'']
        assert status == ServerStatus.DATA_READY

        # Disconnect client
        client.disconnect()
        status, received_data = server.wait_for_activity()[0]
        assert status == ServerStatus.CLIENT_DISCONNECTED
        assert received_data is None

        # Timeout is reached
        status, received_data = server.wait_for_activity()[0]
        assert status == ServerStatus.NOTHING and received_data is None

    def test_send_data(self, serverandclient):
        """
        Tests the `send_data()` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient
        data, _ = self.generate_byte_data()
        assert client.send_data(data) is True

    def test_receive_data(self, serverandclient):
        """
        Tests the `receive_data()` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient
        data, _ = self.generate_byte_data()

        # Not initialized on server side
        with pytest.raises(AttributeError):
            server.receive_data(None, None)

        # Data is sent
        server.accept_client(server.serversocket, None)
        assert client.send_data(data) is True
        status, received_data = server.receive_data(None, None)
        assert status is ServerStatus.DATA_READY
        assert [data] == received_data

        # Client disconnected
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

    def test_collect_messages(self):
        """
        Tests the `collect_messages()` method.
        """
        # Valid data
        protocol = self.initprotocol()
        assert not(protocol.collecteddata)
        data, answer = self.generate_byte_data()
        status, output = protocol.collect_messages(data)
        assert output == answer and status == ServerStatus.DATA_READY

        # Empty data
        protocol = self.initprotocol()
        assert not(protocol.collecteddata)
        status, output = protocol.collect_messages(b'')
        assert output is None and status == ServerStatus.NOTHING

        # Wrong amount of bytes to be read
        data = (10*4).to_bytes(4, 'little', signed=False)
        data += (1).to_bytes(4, 'little', signed=False)
        status, output = protocol.collect_messages(data)
        assert output is None and status == ServerStatus.NOTHING

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
        data, answer = self.generate_byte_data()

        for i in range(10):
            # Send the data
            client_out = client.wait_send(data)
            assert client_out == len(data)

            # Receive data
            server_status, server_data = server.receive_data(None, None)
            assert server_status == ServerStatus.DATA_READY
            assert server_data == answer

    def test_send_message(self, serverandclient):
        """
        Tests the `send_message()` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient
        server.accept_client(server.serversocket, None)
        data, _ = self.generate_byte_data()

        assert client.send_message(MessageType.DATA, data=data) is True
        assert server.send_message(MessageType.DATA, data=data) is True
        client.disconnect()
        with pytest.raises(ConnectionResetError):
            server.send_message(MessageType.OK, data=b'')

    @pytest.mark.parametrize('message,expected', [
        ((MessageType.OK, b''), (True, b'')),
        ((MessageType.ERROR, b''), (False, None)),
        ((MessageType.DATA, b''), (False, None)),
        ((MessageType.MODEL, b''), (False, None)),
        ((MessageType.PROCESS, b''), (False, None)),
        ((MessageType.OUTPUT, b''), (False, None)),
        ((MessageType.STATS, b''), (False, None)),
        ((MessageType.QUANTIZATION, b''), (False, None)),
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
        client.send_message(*message)
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
            client.receive_confirmation()
            client.send_message(MessageType.OK)
            output = client.upload_input(data)
            shared_list.append(output)

        server.accept_client(server.serversocket, None)
        data, _ = self.generate_byte_data()
        shared_list = (multiprocessing.Manager()).list()
        thread = multiprocessing.Process(target=upload,
                                         args=(client, data, shared_list))

        thread.start()
        server.send_message(MessageType.OK)
        assert server.receive_confirmation()[0] is True
        status, message = server.wait_for_activity()[0]
        message_status, received_data = server.parse_message(message[0])
        server.send_message(MessageType.OK, b'')
        thread.join()
        assert status == ServerStatus.DATA_READY
        assert received_data == data
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
        data, _ = self.generate_byte_data()
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
            output = server.send_message(MessageType.OK, b'')
            shared_list.append(output)

        shared_list = (multiprocessing.Manager()).list()
        thread_receive = multiprocessing.Process(target=receive_model,
                                                 args=(server, shared_list))
        server.accept_client(server.serversocket, None)
        thread_receive.start()
        assert client.upload_model(path) is True
        thread_receive.join()
        answer = [(MessageType.MODEL).to_bytes() + data]
        assert shared_list[0] == (ServerStatus.DATA_READY, answer)
        assert shared_list[1] is True

    def test_upload_quantization_details(self, serverandclient, tmpfolder):
        """
        Tests the `upload_quantization_details()` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        tmpfolder : Path
            Fixture to get folder for quantization_details.
        """
        server, client = serverandclient
        quantization_details = {1: 'one', 2: 'two', 3: 'three'}
        path = tmpfolder / uuid.uuid4().hex
        with open(path, 'w') as file:
            json.dump(quantization_details, file)

        def receive_quant(server: RuntimeProtocol, shared_list: list):
            """
            Receives quantization details.

            Parameters
            ----------
            server : RuntimeProtocol
                Initialized RuntimeProtocol server
            shared_list : List
                Shared list to append received quantization details
            """
            status, received = server.receive_data(None, None)
            shared_list.append((status, received))
            output = server.send_message(MessageType.OK, b'')
            shared_list.append(output)

        shared_list = (multiprocessing.Manager()).list()
        args = (server, shared_list)
        thread_receive = multiprocessing.Process(target=receive_quant,
                                                 args=args)
        server.accept_client(server.serversocket, None)

        thread_receive.start()
        assert client.upload_quantization_details(path) is True
        thread_receive.join()

        status, received_data = shared_list[0]
        assert shared_list[1] is True and status == ServerStatus.DATA_READY
        message, json_file = received_data[0][:2], received_data[0][2:]
        assert message == (MessageType.QUANTIZATION).to_bytes()
        assert json_file.decode() == json.dumps(quantization_details)

    def test_download_output(self, serverandclient):
        """
        Tests the `download_output()` method.

        Parameters
        ----------
        serverandclient : Tuple[RuntimeProtocol, RuntimeProtocol]
            Fixture to get initialized server and client
        """
        server, client = serverandclient
        data, _ = self.generate_byte_data()
        server.accept_client(server.serversocket, None)

        assert server.send_message(MessageType.OK, data) is True
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
            client.receive_confirmation()
            client.send_message(MessageType.OK)
            output = client.download_statistics()
            shared_list.append(output)

        shared_list = (multiprocessing.Manager()).list()
        args = (client, shared_list)
        thread_send = multiprocessing.Process(target=download_stats, args=args)
        thread_send.start()

        server.send_message(MessageType.OK)
        assert server.receive_confirmation()[0] is True
        status, message = server.wait_for_activity()[0]
        assert status == ServerStatus.DATA_READY
        message_type, message = server.parse_message(message[0])
        if message_type == MessageType.STATS and message == b'':
            to_send = json.dumps(data).encode()
            output = server.send_message(MessageType.OK, to_send)
            assert output is True
        thread_send.join()
        assert isinstance(shared_list[0], Measurements)
        assert shared_list[0].data == data

    def test_parse_message(self):
        """
        Tests the `parse_message()` method.
        """
        data, _ = self.generate_byte_data()
        protocol = self.initprotocol()

        for i in range(8):
            tmp = (i).to_bytes(2, byteorder='little', signed=False) + data
            mt, output_data = protocol.parse_message(tmp)
            assert mt.value == i and output_data == data
        with pytest.raises(ValueError):
            tmp = (100).to_bytes(2, byteorder='little', signed=False)
            protocol.parse_message(tmp + data)

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
        assert client.send_message(MessageType.OK) is True
        assert server.send_message(MessageType.OK) is True
        client.disconnect()
        with pytest.raises(OSError):
            client.send_message(MessageType.OK)
        with pytest.raises(ConnectionResetError):
            server.send_message(MessageType.OK)
        server.disconnect()
        with pytest.raises(OSError):
            server.send_message(MessageType.OK)

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

        def send_confirmation(client):
            client.send_message(MessageType.OK)
            client.send_message(MessageType.OK)

        def send_reject_first(client):
            client.send_message(MessageType.ERROR)

        def send_reject_second(client):
            client.send_message(MessageType.OK)
            client.send_message(MessageType.ERROR)

        functions = (send_confirmation, send_reject_first, send_reject_second)
        expected = (True, False, False)
        args = (client, )

        for function, expected in zip(functions, expected):
            thread_send = multiprocessing.Process(target=function, args=args)
            thread_send.start()
            assert server.request_processing() is expected
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
        message_type, message = client.parse_message(message[0])
        assert message_type == MessageType.ERROR and message == b''

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
        data, _ = self.generate_byte_data()
        server.request_success(data)
        status, message = client.receive_data(None, None)
        assert status == ServerStatus.DATA_READY
        message_type, message = client.parse_message(message[0])
        assert message_type == MessageType.OK and message == data
