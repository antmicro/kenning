from runtimeprotocolbase import RuntimeProtocolTests
from kenning.runtimeprotocols.network import NetworkProtocol
from kenning.core.runtimeprotocol import RuntimeProtocol, ServerStatus
from kenning.core.runtimeprotocol import MessageType
from typing import Tuple, List
import pytest
import random
import socket
import multiprocessing


class TestNetworkProtocol(RuntimeProtocolTests):
    runtimeprotocolcls = NetworkProtocol
    host = ''
    port = 1235

    def generate_byte_data(self) -> Tuple[bytes, List[bytes]]:
        """
        Generates correct data for test.

        Returns
        -------
        Tuple[bytes, List[bytes]]:
            A tuple containing bytes stream and expected output
        """
        data = bytes()
        answer = list()
        for i in range(random.randint(1, 10)):
            times = random.randint(1, 10)
            answer.append(bytes())
            tmp_order = bytes()
            for j in range(times):
                number = (random.randint(1, 4294967295))
                num_bytes = number.to_bytes(4, byteorder='little',
                                            signed=False)
                tmp_order += num_bytes
                answer[i] += num_bytes
            length = len(tmp_order)
            length = length.to_bytes(4, byteorder='little', signed=False)
            data += length + tmp_order
        return data, answer

    def test_wait_for_activity(self):
        server = self.initprotocol()
        server.initialize_server()

        # Timeout is reached
        status, received_data = server.wait_for_activity()[0]
        assert status == ServerStatus.NOTHING and received_data is None

        # Connect via Client
        client = self.initprotocol()
        client.initialize_client()
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

        # Send fully empty message
        class InvalidMessage:
            def to_bytes(self):
                return b''

        client.send_message(InvalidMessage(), b'')
        status, received_data = server.wait_for_activity()[0]
        assert received_data == [b'']
        assert status == ServerStatus.DATA_INVALID

        # Disconnect with client
        client.disconnect()
        status, received_data = server.wait_for_activity()[0]
        assert status == ServerStatus.CLIENT_DISCONNECTED
        assert received_data is None
        server.disconnect()

    def test_accept_client(self):
        def connect():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.host, self.port))
            s.close()

        def run_test(protocol: RuntimeProtocol):
            output: ServerStatus
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((self.host, self.port))
                s.listen(1)
                multiprocessing.Process(target=connect).start()
                output = protocol.accept_client(s, None)
                s.shutdown(socket.SHUT_RDWR)
            return output

        # Test there's already established connection
        protocol = self.initprotocol()
        protocol.socket = True
        assert run_test(protocol)[0] == ServerStatus.CLIENT_IGNORED

        # Test there was no connection yet
        protocol = self.initprotocol()
        assert run_test(protocol)[0] == ServerStatus.CLIENT_CONNECTED

        # Test attribute
        protocol = self.initprotocol()
        with pytest.raises(AttributeError):
            protocol.accept_client(None, None)

    def test_collect_messages(self):
        # valid data
        protocol = self.initprotocol()
        assert not(protocol.collecteddata)
        data, answer = self.generate_byte_data()
        status, output = protocol.collect_messages(data)
        assert output == answer and status == ServerStatus.DATA_READY

        # empty data
        protocol = self.initprotocol()
        assert not(protocol.collecteddata)
        status, output = protocol.collect_messages(b'')
        assert output is None and status == ServerStatus.NOTHING

        # wrong amount of bytes to be read
        data = (10*4).to_bytes(4, 'little', signed=False)
        data += (1).to_bytes(4, 'little', signed=False)
        status, output = protocol.collect_messages(data)
        assert output is None and status == ServerStatus.NOTHING

    def test_wait_send(self):
        server = self.initprotocol()
        client = self.initprotocol()
        server.initialize_server()
        client.initialize_client()
        server.accept_client(server.serversocket, None)
        data, answer = self.generate_byte_data()

        for i in range(10):
            # Send the data
            client_out = client.wait_send(data)
            assert client_out == len(data)

            # Recieve data
            server_status, server_data = server.receive_data(None, None)
            assert server_status == ServerStatus.DATA_READY
            assert server_data == answer

        client.disconnect()
        server.disconnect()

    def test_send_message(self):
        server = self.initprotocol()
        client = self.initprotocol()
        server.initialize_server()
        client.initialize_client()
        server.accept_client(server.serversocket, None)
        data, _ = self.generate_byte_data()
        assert client.send_message(MessageType.DATA, data=data) is True
        assert server.send_message(MessageType.DATA, data=data) is True
        client.disconnect()
        with pytest.raises(ConnectionResetError):
            server.send_message(MessageType.OK, data=b'')

        server.disconnect()

    def test_receive_confirmation(self):

        def confirm(server: NetworkProtocol, return_list: list):
            """
            Waits for message and appends output to provided shared list.
            """
            output = server.receive_confirmation()
            return_list.append(output)

        manager = multiprocessing.Manager()
        shared_list = manager.list()
        server = self.initprotocol()
        client = self.initprotocol()
        server.initialize_server()
        client.initialize_client()
        server.accept_client(server.serversocket, None)

        cases = (((MessageType.OK, b''), (True, b'')),
                 ((MessageType.ERROR, b''), (False, None)),
                 ((MessageType.DATA, b''), (False, None)),
                 ((MessageType.MODEL, b''), (False, None)),
                 ((MessageType.PROCESS, b''), (False, None)),
                 ((MessageType.OUTPUT, b''), (False, None)),
                 ((MessageType.STATS, b''), (False, None)),
                 ((MessageType.QUANTIZATION, b''), (False, None)),
                 )

        # Check for every presented MessageType
        for message, expected in cases:
            thread = multiprocessing.Process(target=confirm,
                                             args=(server, shared_list))
            thread.start()
            client.send_message(*message)
            thread.join()
            assert shared_list[-1] == expected

        # Check if client is disconnected
        client.disconnect()
        output = server.receive_confirmation()
        assert output == (False, None)

        server.disconnect()
