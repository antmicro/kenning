from runtimeprotocolbase import RuntimeProtocolTests
from kenning.runtimeprotocols.network import NetworkProtocol
from kenning.core.runtimeprotocol import RuntimeProtocol, ServerStatus
import pytest
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

    @pytest.mark.xfail
    def test_collect_messages(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_wait_send(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_send_message(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_recieve_confirmation(self):
        raise NotImplementedError
