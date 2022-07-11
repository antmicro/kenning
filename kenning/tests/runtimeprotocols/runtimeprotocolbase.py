from kenning.core.runtimeprotocol import RuntimeProtocol
from kenning.core.measurements import Measurements
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
        protocol = self.runtimeprotocolcls(*args, **kwargs)
        return protocol

    def test_initialize_server(self):
        raise NotImplementedError

    def test_initialize_client(self):
        raise NotImplementedError

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
