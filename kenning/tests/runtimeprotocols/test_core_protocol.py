# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from kenning.core.runtimeprotocol import MessageType, RuntimeProtocol


@pytest.mark.fast
class TestMessageType:
    def test_to_bytes(self):
        """
        Test converting message to bytes.
        """
        byte_num = (1).to_bytes(2, 'little', signed=False)
        assert MessageType.ERROR.to_bytes() == byte_num

    def test_from_bytes(self):
        """
        Test converting message from bytes.
        """
        byte_num = (1).to_bytes(2, 'little', signed=False)
        assert MessageType.ERROR == MessageType.from_bytes(byte_num, 'little')


@pytest.mark.fast
class TestCoreRuntimeProtocol:
    def init_protocol(self) -> RuntimeProtocol:
        """
        Initializes protocol object.

        Returns
        -------
        RuntimeProtocol:
            Initialized protocol object
        """
        return RuntimeProtocol()

    @pytest.fixture
    def server_and_client(self):
        """
        Initializes server and client.

        Returns
        -------
        Tuple[RuntimeProtocol, RuntimeProtocol] :
            A tuple containing initialized server and client objects
        """
        server = self.init_protocol()
        if server.initialize_server() is False:
            pytest.fail('Server initialization failed')

        client = self.init_protocol()
        if client.initialize_client() is False:
            pytest.fail('Client initialization failed')

        yield server, client
        client.disconnect()
        server.disconnect()

    def test_download_statistics(self):
        """
        Tests the `RuntimeProtocol.download_statistics()` method.
        """
        client = self.init_protocol()
        with pytest.raises(NotImplementedError):
            client.download_statistics()

    def test_initialize_server(self):
        protocol = self.init_protocol()
        with pytest.raises(NotImplementedError):
            protocol.initialize_server()

    def test_initialize_client(self):
        protocol = self.init_protocol()
        with pytest.raises(NotImplementedError):
            protocol.initialize_client()

    def test_request_processing(self):
        protocol = self.init_protocol()
        with pytest.raises(NotImplementedError):
            protocol.request_processing()

    def test_request_success(self):
        protocol = self.init_protocol()
        with pytest.raises(NotImplementedError):
            protocol.request_success()

    def test_request_failure(self):
        protocol = self.init_protocol()
        with pytest.raises(NotImplementedError):
            protocol.request_failure()

    def test_disconnect(self):
        protocol = self.init_protocol()
        with pytest.raises(NotImplementedError):
            protocol.disconnect()
