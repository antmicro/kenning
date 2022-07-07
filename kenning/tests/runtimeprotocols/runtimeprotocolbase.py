from kenning.runtimeprotocols.network import NetworkProtocol
from kenning.core.runtimeprotocol import RuntimeProtocol
from kenning.core.measurements import Measurements
import pytest


@pytest.mark.fast
class RuntimeProtocolTests:

    @pytest.fixture
    def server(self):
        server = self.initprotocol()
        server.initialize_server()
        yield server
        server.disconnect()

    @pytest.fixture
    def client(self):
        client = self.initprotocol()
        client.initialize_client()
        yield client
        client.disconnect()

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
        server = self.initprotocol()
        assert server.initialize_server() is True
        with pytest.raises(OSError) as execinfo:
            second_server = self.initprotocol()
            second_server.initialize_server()
        assert 'Address already in use' in str(execinfo.value)
        server.disconnect()

    def test_initialize_client(self):
        client = self.initprotocol()
        with pytest.raises(ConnectionRefusedError):
            client.initialize_client()
        server = self.initprotocol()
        server.initialize_server()
        client.initialize_client()

        client.disconnect()
        server.disconnect()

    def test_wait_for_activity(self):
        raise NotImplementedError

    def test_send_data(self):
        raise NotImplementedError

    def test_receive_data(self):
        raise NotImplementedError

    def test_upload_input(self):
        raise NotImplementedError

    def test_upload_model(self):
        raise NotImplementedError

    def test_upload_quantization_details(self):
        raise NotImplementedError

    def test_download_output(self):
        raise NotImplementedError

    def test_download_statistics(self):
        client = self.initprotocol()
        assert isinstance(client.download_statistics(), Measurements)

    def test_request_processing(self):
        raise NotImplementedError

    def test_request_success(self):
        raise NotImplementedError

    def test_request_failure(self):
        raise NotImplementedError

    def test_parse_message(self):
        raise NotImplementedError

    def test_disconnect(self):
        raise NotImplementedError
