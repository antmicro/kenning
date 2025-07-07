# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext as does_not_raise
from typing import Optional

import pytest

from kenning.protocols.message import (
    HEADER_SIZE,
    FlagName,
    Flags,
    FlowControlFlags,
    Message,
    MessageType,
)
from kenning.tests.utils.test_serializable import serializables_equal


class TestFlags:
    @pytest.mark.parametrize(
        "example_flags",
        [
            # Empty flag dict (default values for all flags)
            ({}),
            # Some flags set
            (
                {
                    FlagName.FAIL: 1,
                    FlagName.SPEC_FLAG_2: 1,
                    FlagName.HAS_PAYLOAD: 0,
                    FlagName.IS_HOST_MESSAGE: 1,
                }
            ),
            # All flags set
            (
                {
                    FlagName.SUCCESS: 1,
                    FlagName.FAIL: 1,
                    FlagName.HAS_PAYLOAD: 1,
                    FlagName.IS_HOST_MESSAGE: 1,
                    FlagName.FIRST: 1,
                    FlagName.LAST: 1,
                    FlagName.SPEC_FLAG_2: 1,
                    FlagName.SPEC_FLAG_1: 1,
                    FlagName.SPEC_FLAG_3: 1,
                    FlagName.SPEC_FLAG_4: 1,
                }
            ),
        ],
    )
    def test_init(self, example_flags):
        flags_object = Flags(example_flags)
        for field, _, _ in flags_object.serializable_fields:
            if field in example_flags.keys():
                # If the flag was in the Dict, we check if it was assigned
                assert example_flags[field] == getattr(
                    flags_object, str(field)
                )
            else:
                # Otherwise, we check if the default value was set.
                assert 0 == getattr(flags_object, str(field))


class TestMessage:
    @pytest.mark.parametrize(
        "bytestream,message,size,valid_checkusm",
        [
            # Valid transmission message with byte payload.
            (
                b"\xC7\x45\x2D\xE0\x05\x00\x00\x00\x01\x02\x03\x04\x05\x44\x40\x00\x10\x01\x56\x89\xC0\x44\x40\x00\x10\x01\x56\x89\xC0",
                Message(
                    MessageType.IO_SPEC,
                    b"\x01\x02\x03\x04\x05",
                    FlowControlFlags.TRANSMISSION,
                    Flags(
                        {
                            FlagName.SUCCESS: 1,
                            FlagName.FAIL: 0,
                            FlagName.IS_HOST_MESSAGE: 1,
                            FlagName.FIRST: 0,
                            FlagName.LAST: 1,
                            FlagName.SPEC_FLAG_1: 0,
                            FlagName.SPEC_FLAG_2: 1,
                            FlagName.SPEC_FLAG_3: 1,
                            FlagName.SPEC_FLAG_4: 1,
                        }
                    ),
                ),
                13,
                True,
            ),
            # Transmission message with byte payload, with invalid checksum.
            (
                b"\xC7\x47\x2D\xE0\x05\x00\x00\x00\x01\x02\x03\x04\x05\x44\x40\x00\x10\x01\x56\x89\xC0\x44\x40\x00\x10\x01\x56\x89\xC0",
                Message(
                    MessageType.IO_SPEC,
                    b"\x01\x02\x03\x04\x05",
                    FlowControlFlags.TRANSMISSION,
                    Flags(
                        {
                            FlagName.SUCCESS: 1,
                            FlagName.FAIL: 0,
                            FlagName.IS_HOST_MESSAGE: 1,
                            FlagName.FIRST: 0,
                            FlagName.LAST: 1,
                            FlagName.SPEC_FLAG_1: 0,
                            FlagName.SPEC_FLAG_2: 1,
                            FlagName.SPEC_FLAG_3: 1,
                            FlagName.SPEC_FLAG_4: 1,
                        }
                    ),
                ),
                13,
                False,
            ),
            # Transmission message with a single bit error in the payload
            (
                b"\xC7\x45\x2D\xE0\x05\x00\x00\x00\x01\x02\x02\x04\x05\x44\x40\x00\x10\x01\x56\x89\xC0\x44\x40\x00\x10\x01\x56\x89\xC0",
                Message(
                    MessageType.IO_SPEC,
                    b"\x01\x02\x02\x04\x05",  # Error in the 3rd byte
                    FlowControlFlags.TRANSMISSION,
                    Flags(
                        {
                            FlagName.SUCCESS: 1,
                            FlagName.FAIL: 0,
                            FlagName.IS_HOST_MESSAGE: 1,
                            FlagName.FIRST: 0,
                            FlagName.LAST: 1,
                            FlagName.SPEC_FLAG_1: 0,
                            FlagName.SPEC_FLAG_2: 1,
                            FlagName.SPEC_FLAG_3: 1,
                            FlagName.SPEC_FLAG_4: 1,
                        }
                    ),
                ),
                13,
                False,
            ),
            # Transmission message with an error in message identifier
            (
                b"\xCF\x45\x2D\xE0\x05\x00\x00\x00\x01\x02\x03\x04\x05\x44\x40\x00\x10\x01\x56\x89\xC0\x44\x40\x00\x10\x01\x56\x89\xC0",
                Message(
                    None,  # Error in message type
                    b"\x01\x02\x03\x04\x05",
                    FlowControlFlags.TRANSMISSION,
                    Flags(
                        {
                            FlagName.SUCCESS: 1,
                            FlagName.FAIL: 0,
                            FlagName.IS_HOST_MESSAGE: 1,
                            FlagName.FIRST: 0,
                            FlagName.LAST: 1,
                            FlagName.SPEC_FLAG_1: 0,
                            FlagName.SPEC_FLAG_2: 1,
                            FlagName.SPEC_FLAG_3: 1,
                            FlagName.SPEC_FLAG_4: 1,
                        }
                    ),
                ),
                13,
                False,
            ),
            # Transmission message with an error in flags
            (
                b"\xC7\x45\x2C\xE0\x05\x00\x00\x00\x01\x02\x03\x04\x05\x44\x40\x00\x10\x01\x56\x89\xC0\x44\x40\x00\x10\x01\x56\x89\xC0",
                Message(
                    MessageType.IO_SPEC,
                    b"\x01\x02\x03\x04\x05",
                    FlowControlFlags.TRANSMISSION,
                    Flags(
                        {
                            FlagName.SUCCESS: 0,  # Error in the flag
                            FlagName.FAIL: 0,
                            FlagName.IS_HOST_MESSAGE: 1,
                            FlagName.FIRST: 0,
                            FlagName.LAST: 1,
                            FlagName.SPEC_FLAG_1: 0,
                            FlagName.SPEC_FLAG_2: 1,
                            FlagName.SPEC_FLAG_3: 1,
                            FlagName.SPEC_FLAG_4: 1,
                        }
                    ),
                ),
                13,
                False,
            ),
            # Transmission message with an error in the size field
            (
                b"\xC7\x45\x2D\xE0\x07\x00\x00\x00\x01\x02\x03\x04\x05\x44\x40\x00\x10\x01\x56\x89\xC0\x44\x40\x00\x10\x01\x56\x89\xC0",
                Message(
                    MessageType.IO_SPEC,
                    b"\x01\x02\x03\x04\x05\x44\x40",
                    FlowControlFlags.TRANSMISSION,
                    Flags(
                        {
                            FlagName.SUCCESS: 1,
                            FlagName.FAIL: 0,
                            FlagName.IS_HOST_MESSAGE: 1,
                            FlagName.FIRST: 0,
                            FlagName.LAST: 1,
                            FlagName.SPEC_FLAG_1: 0,
                            FlagName.SPEC_FLAG_2: 1,
                            FlagName.SPEC_FLAG_3: 1,
                            FlagName.SPEC_FLAG_4: 1,
                        }
                    ),
                ),
                15,
                False,
            ),
            # Too short bytestream - message with payload
            (
                b"\xC7\x45\x2D\xE0\x05\x00\x00\x00\x01\x02",
                None,
                0,
                None,
            ),
            # Too short bytestream - message without
            (
                b"\xC5\xCF\x21\x60\x00\x00\x00",
                None,
                0,
                None,
            ),
        ],
    )
    def test_from_bytes(self, bytestream, message, size, valid_checkusm):
        (
            deserialized_message,
            deserialized_size,
            deserialized_valid_checksum,
        ) = Message.from_bytes(bytestream)
        assert size == deserialized_size
        assert valid_checkusm == deserialized_valid_checksum
        assert type(message) == type(deserialized_message)
        if type(message) is Message:
            assert message.message_type == deserialized_message.message_type
            assert (
                message.flow_control_flags
                == deserialized_message.flow_control_flags
            )
            serializables_equal(message.flags, deserialized_message.flags)
            assert message.message_size == deserialized_message.message_size
            assert message.payload == deserialized_message.payload

    @pytest.mark.parametrize(
        "message_type,payload,flow_control,flags,correct_message_bytestream,expectation",
        [
            # Valid TRANSMISSION message with payload
            (
                MessageType.IO_SPEC,
                b"\x01\x02\x03\x04\x05",
                FlowControlFlags.TRANSMISSION,
                Flags(
                    {
                        FlagName.SUCCESS: 1,
                        FlagName.FAIL: 0,
                        FlagName.IS_HOST_MESSAGE: 1,
                        FlagName.FIRST: 0,
                        FlagName.LAST: 1,
                        FlagName.SPEC_FLAG_1: 0,
                        FlagName.SPEC_FLAG_2: 1,
                        FlagName.SPEC_FLAG_3: 1,
                        FlagName.SPEC_FLAG_4: 1,
                    }
                ),
                b"\xC7\x45\x2D\xE0\x05\x00\x00\x00\x01\x02\x03\x04\x05",
                does_not_raise(),
            ),
            # Valid TRANSMISSION message without payload
            (
                MessageType.OUTPUT,
                None,
                FlowControlFlags.TRANSMISSION,
                Flags(
                    {
                        FlagName.SUCCESS: 1,
                        FlagName.FAIL: 0,
                        FlagName.IS_HOST_MESSAGE: 0,
                        FlagName.FIRST: 0,
                        FlagName.LAST: 1,
                        FlagName.SPEC_FLAG_1: 0,
                        FlagName.SPEC_FLAG_2: 1,
                        FlagName.SPEC_FLAG_3: 1,
                        FlagName.SPEC_FLAG_4: 0,
                    }
                ),
                b"\xC5\xCF\x21\x60\x00\x00\x00\x00",
                does_not_raise(),
            ),
            # Invalid message (non-serialized payload)
            (
                MessageType.OUTPUT,
                31,
                FlowControlFlags.REQUEST_RETRANSMIT,
                Flags(
                    {
                        FlagName.SUCCESS: 1,
                        FlagName.FAIL: 0,
                        FlagName.IS_HOST_MESSAGE: 0,
                        FlagName.FIRST: 0,
                        FlagName.LAST: 1,
                        FlagName.SPEC_FLAG_1: 0,
                        FlagName.SPEC_FLAG_2: 1,
                        FlagName.SPEC_FLAG_3: 1,
                        FlagName.SPEC_FLAG_4: 1,
                    }
                ),
                b"\x45\xD0\x21\xE0\x1F\x00\x00\x00",
                pytest.raises(ValueError),
            ),
        ],
    )
    def test_compatibility(
        self,
        message_type: MessageType,
        payload: Optional["bytes"] | int,
        flow_control: FlowControlFlags,
        flags: Flags,
        correct_message_bytestream: bytes,
        expectation,
    ):
        with expectation:
            message = Message(message_type, payload, flow_control, flags)
            serialized_message = message.to_bytes()
            deserialized_message, size, checksum_valid = Message.from_bytes(
                serialized_message
            )
            assert Message == type(deserialized_message)
            if flags.has_payload:
                assert len(payload) + HEADER_SIZE == size
            else:
                assert HEADER_SIZE == size

            assert correct_message_bytestream == serialized_message

            assert message.message_type == deserialized_message.message_type
            assert message.message_size == deserialized_message.message_size
            serializables_equal(message.flags, deserialized_message.flags)
            assert (
                message.flow_control_flags
                == deserialized_message.flow_control_flags
            )
            if message.flags.has_payload:
                assert message.payload == deserialized_message.payload
                assert payload == deserialized_message.payload
            assert message.message_size == deserialized_message.message_size
            assert len(correct_message_bytestream) == size
