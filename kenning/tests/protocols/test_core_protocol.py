# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from kenning.core.protocol import MessageType, Protocol


@pytest.mark.fast
class TestMessageType:
    def test_to_bytes(self):
        """
        Test converting message to bytes.
        """
        byte_num = (1).to_bytes(2, "little", signed=False)
        assert MessageType.ERROR.to_bytes() == byte_num

    def test_from_bytes(self):
        """
        Test converting message from bytes.
        """
        byte_num = (1).to_bytes(2, "little", signed=False)
        assert MessageType.ERROR == MessageType.from_bytes(byte_num, "little")


@pytest.mark.fast
class TestCoreProtocol:
    @patch.multiple(Protocol, __abstractmethods__=set())
    def init_protocol(self) -> Protocol:
        """
        Initializes protocol object.

        Returns
        -------
        TestProtocol
            Initialized protocol object
        """
        return Protocol()

    @pytest.fixture
    def server_and_client(self):
        """
        Initializes server and client.

        Returns
        -------
        Tuple[Protocol, Protocol]
            A tuple containing initialized server and client objects
        """
        server = self.init_protocol()
        if server.initialize_server() is False:
            pytest.fail("Server initialization failed")

        client = self.init_protocol()
        if client.initialize_client() is False:
            pytest.fail("Client initialization failed")

        yield server, client
        client.disconnect()
        server.disconnect()
