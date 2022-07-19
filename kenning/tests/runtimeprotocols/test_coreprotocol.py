from kenning.core.measurements import Measurements
from kenning.core.runtimeprotocol import RuntimeProtocol, MessageType
from typing import Tuple, List
import pytest
import random


@pytest.mark.fast
class TestMessageType:
    def test_to_bytes(self):
        byte_num = (1).to_bytes(2, 'little', signed=False)
        assert MessageType.ERROR.to_bytes() == byte_num

    def test_from_bytes(self):
        byte_num = (1).to_bytes(2, 'little', signed=False)
        assert MessageType.ERROR == MessageType.from_bytes(byte_num, 'little')


@pytest.mark.fast
class TestCoreRuntimeProtocol:
    runtimeprotocolcls = RuntimeProtocol

    def initprotocol(self) -> RuntimeProtocol:
        """
        Initializes protocol object.

        Returns
        -------
        RuntimeProtocol:
            Initialized protocol object
        """
        protocol = self.runtimeprotocolcls()
        return protocol

    @pytest.fixture
    def serverandclient(self):
        """
        Initializes server and client.

        Returns
        -------
        Tuple[RuntimeProtocol, RuntimeProtocol] :
            A tuple containing initialized server and client objects
        """
        server = self.initprotocol()
        client = self.initprotocol()
        server.initialize_server()
        client.initialize_client()
        yield server, client
        client.disconnect()
        server.disconnect()

    def generate_byte_data(self) -> Tuple[bytes, List[bytes]]:
        """
        Generates random data in byte format for tests.

        Returns
        -------
        Tuple[bytes, List[bytes]] :
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

    def test_download_statistics(self):
        """
        Tests the `RuntimeProtocol.download_statistics()` method.
        """
        client = self.initprotocol()
        assert isinstance(client.download_statistics(), Measurements)

    def test_initialize_server(self):
        protocol = self.initprotocol()
        with pytest.raises(NotImplementedError):
            protocol.initialize_server()

    def test_initialize_client(self):
        protocol = self.initprotocol()
        with pytest.raises(NotImplementedError):
            protocol.initialize_client()

    def test_wait_for_activity(self):
        protocol = self.initprotocol()
        with pytest.raises(NotImplementedError):
            protocol.wait_for_activity()

    def test_send_data(self):
        protocol = self.initprotocol()
        with pytest.raises(NotImplementedError):
            protocol.send_data(b'')

    def test_receive_data(self):
        protocol = self.initprotocol()
        with pytest.raises(NotImplementedError):
            protocol.receive_data()

    def test_upload_input(self):
        protocol = self.initprotocol()
        with pytest.raises(NotImplementedError):
            protocol.upload_input(b'')

    def test_upload_model(self, tmpfolder):
        protocol = self.initprotocol()
        with pytest.raises(NotImplementedError):
            protocol.upload_model(tmpfolder)

    def test_upload_quantization_details(self, tmpfolder):
        protocol = self.initprotocol()
        with pytest.raises(NotImplementedError):
            protocol.upload_quantization_details(tmpfolder)

    def test_download_output(self):
        protocol = self.initprotocol()
        with pytest.raises(NotImplementedError):
            protocol.download_output()

    def test_request_processing(self):
        protocol = self.initprotocol()
        with pytest.raises(NotImplementedError):
            protocol.request_processing()

    def test_request_success(self):
        protocol = self.initprotocol()
        with pytest.raises(NotImplementedError):
            protocol.request_success()

    def test_request_failure(self):
        protocol = self.initprotocol()
        with pytest.raises(NotImplementedError):
            protocol.request_failure()

    def test_parse_message(self):
        protocol = self.initprotocol()
        with pytest.raises(NotImplementedError):
            protocol.parse_message(bytes())

    def test_disconnect(self):
        protocol = self.initprotocol()
        with pytest.raises(NotImplementedError):
            protocol.disconnect()
