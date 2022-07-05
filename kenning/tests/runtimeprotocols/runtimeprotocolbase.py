from kenning.runtimeprotocols.network import NetworkProtocol
from kenning.core.runtimeprotocol import RuntimeProtocol
import pytest


@pytest.mark.fast
class RuntimeProtocolTests:
    def initprotocol(self, *args, **kwargs) -> RuntimeProtocol:
        """
        Initializes protocol object.

        Returns
        -------
        RuntimeProtocol:
            Initialized protocol object
        """
        protocol = NetworkProtocol(self.host, self.port, *args, **kwargs)
        return protocol

    def test_initialize_server(self):
        protocol = self.initprotocol()
        protocol.initialize_server()

    def test_initialize_client(self):
        protocol = self.initprotocol()
        protocol.initialize_client()

    @pytest.mark.xfail()
    def test_wait_for_activity(self):
        assert 0

    @pytest.mark.xfail()
    def test_send_data(self):
        assert 0

    @pytest.mark.xfail()
    def test_receive_data(self):
        assert 0

    @pytest.mark.xfail()
    def test_upload_input(self):
        assert 0

    @pytest.mark.xfail()
    def test_upload_model(self):
        assert 0

    @pytest.mark.xfail()
    def test_upload_quantization_details(self):
        assert 0

    @pytest.mark.xfail()
    def test_request_processing(self):
        assert 0

    @pytest.mark.xfail()
    def test_download_output(self):
        assert 0

    @pytest.mark.xfail()
    def test_download_statistics(self):
        assert 0

    @pytest.mark.xfail()
    def test_request_success(self):
        assert 0

    @pytest.mark.xfail()
    def test_request_failure(self):
        assert 0

    @pytest.mark.xfail()
    def test_parse_message(self):
        assert 0

    @pytest.mark.xfail()
    def test_disconnect(self):
        assert 0


@pytest.mark.fast
class TestCheckRequest:
    @pytest.mark.xfail()
    def test_one(self):
        assert 0


@pytest.mark.fast
class TestRequestFailure:
    @pytest.mark.xfail()
    def test_one(self):
        assert 0


@pytest.mark.fast
class TestMessageType:
    @pytest.mark.xfail()
    def test_to_bytes(self):
        assert 0

    @pytest.mark.xfail()
    def test_from_bytes(self):
        assert 0


@pytest.mark.fast
class TestServerStatus:
    @pytest.mark.xfail()
    def test_one(self):
        assert 0
