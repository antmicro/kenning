from runtimeprotocolbase import RuntimeProtocolTests
from kenning.runtimeprotocols.network import NetworkProtocol
import pytest


class TestNetworkProtocol(RuntimeProtocolTests):
    runtimeprotocolcls = NetworkProtocol
    host = '127.0.0.28'
    port = 1234

    @pytest.mark.xfail
    def test_accept_client(self):
        raise NotImplementedError

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
