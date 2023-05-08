# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from kenning.core.runtimeprotocol import RuntimeProtocol, MessageType
from typing import Tuple, List
from time import time
import pytest
import random
import socket


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
    def randomport(self, record_property):
        """
        Sets random free port number within dynamic port range.
        """
        random.seed(time())
        count = 0

        # Check if port is not used
        while True:
            try:
                self.port = random.randint(49152, 65535)
                count += 1
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(('', self.port))
                s.close()
                break
            except OSError:
                if count > 5:
                    pytest.fail('Cannot find free port')
                continue
        record_property('random_port', self.port)

    @pytest.fixture
    def serverandclient(self, randomport):
        """
        Initializes server and client.

        Parameters
        ----------
        randomport : None
            Fixture to set free random port number within dynamic port range.

        Returns
        -------
        Tuple[RuntimeProtocol, RuntimeProtocol] :
            A tuple containing initialized server and client objects
        """
        server = self.initprotocol()
        if server.initialize_server() is False:
            pytest.fail(f'Server initialization failed, port: {self.port}')
        client = self.initprotocol()
        client.initialize_client()

        yield server, client
        client.disconnect()
        server.disconnect()

    def generate_byte_data(self) -> Tuple[bytes, List[bytes]]:
        """
        Generates random data in byte format for tests.

        Returns
        -------
        bytes :
            Byte array of random data
        """
        data = bytes()
        for i in range(random.randint(1, 10)):
            times = random.randint(1, 10)
            tmp_order = bytes()
            for _ in range(times):
                number = random.randint(0, 0xFFFFFFFF)
                num_bytes = number.to_bytes(4, byteorder='little',
                                            signed=False)
                tmp_order += num_bytes
            length = len(tmp_order)
            length = length.to_bytes(4, byteorder='little', signed=False)
            data += length + tmp_order
        return data

    def test_download_statistics(self):
        """
        Tests the `RuntimeProtocol.download_statistics()` method.
        """
        client = self.initprotocol()
        with pytest.raises(NotImplementedError):
            client.download_statistics()

    def test_initialize_server(self):
        protocol = self.initprotocol()
        with pytest.raises(NotImplementedError):
            protocol.initialize_server()

    def test_initialize_client(self):
        protocol = self.initprotocol()
        with pytest.raises(NotImplementedError):
            protocol.initialize_client()

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

    def test_disconnect(self):
        protocol = self.initprotocol()
        with pytest.raises(NotImplementedError):
            protocol.disconnect()
