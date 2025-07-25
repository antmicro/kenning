import pytest

from kenning.protocols.bytes_based_protocol import TransmissionFlag
from kenning.protocols.message import MessageType


class TestTransmissionFlag:
    @pytest.mark.parametrize(
        "flag, message_type, return_value",
        [
            # General purpose flag and message type without special flags.
            (TransmissionFlag.SUCCESS, MessageType.OUTPUT, True),
            # General purpose flag and message type with special flags.
            (TransmissionFlag.SUCCESS, MessageType.IO_SPEC, True),
            # Message type specific flag with valid message type
            (TransmissionFlag.SERIALIZED, MessageType.IO_SPEC, True),
            # Message type specific flag with invalid message type
            (TransmissionFlag.SERIALIZED, MessageType.MODEL, False),
        ],
    )
    def test_for_type(
        self,
        flag: TransmissionFlag,
        message_type: MessageType,
        return_value: bool,
    ):
        assert return_value == flag.for_type(message_type)
