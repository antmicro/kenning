from runtimeprotocolbase import RuntimeProtocolTests
from kenning.runtimeprotocols.network import NetworkProtocol
from kenning.core.runtimeprotocol import RuntimeProtocol, ServerStatus
from typing import Tuple, List
import pytest
import random
import socket
import multiprocessing


class TestNetworkProtocol(RuntimeProtocolTests):
    runtimeprotocolcls = NetworkProtocol
    host = ''
    port = 1235

    def test_accept_client(self):
        def connect():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.host, self.port))
            s.shutdown(socket.SHUT_RDWR)
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

        def generate_byte_data() -> Tuple[bytes, List[bytes]]:
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
                times = random.randint(0, 10)
                data += (4*times).to_bytes(4, byteorder='little', signed=False)
                answer.append(b'')
                for j in range(times):
                    number = (random.randint(0, 4294967295))
                    num_bytes = number.to_bytes(4, byteorder='little',
                                                signed=False)
                    data += num_bytes
                    answer[i] += num_bytes
            return data, answer

        # valid data
        protocol = self.initprotocol()
        assert not(protocol.collecteddata)
        data, answer = generate_byte_data()
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

    @pytest.mark.xfail
    def test_wait_send(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_send_message(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_recieve_confirmation(self):
        raise NotImplementedError
