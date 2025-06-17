# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0


import pytest

from kenning.protocols.bytes_based_protocol import MessageType


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
