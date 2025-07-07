# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import time
from math import ceil
from threading import Event, Thread
from typing import Callable, List, Optional, Tuple
from unittest.mock import patch

import pytest

from kenning.protocols.kenning_protocol import (
    MAX_MESSAGE_PAYLOAD_SIZE,
    FlowControlFlags,
    IncomingRequest,
    IncomingTransmission,
    Listen,
    MessageType,
    OutgoingRequest,
    OutgoingTransmission,
    ProtocolEvent,
    Transmission,
    TransmissionFlag,
)
from kenning.protocols.message import FlagName, Flags, Message
from kenning.utils.event_with_args import EventWithArgs

DEFAULT_MESSAGE_TYPE = MessageType.DATA


@pytest.fixture
def message_type():
    return DEFAULT_MESSAGE_TYPE


@pytest.fixture
@patch.multiple(ProtocolEvent, __abstractmethods__=set())
def protocol_event(message_type: MessageType):
    protocol_event = ProtocolEvent(message_type)
    protocol_event.INITIAL_ACTIVE_STATE = ProtocolEvent.State.NEW
    return protocol_event


# Example transmission of type MODEL (series of 5 messages)
# with validation values (message type, expected payload and
# expected transmission flags).
MODEL_TRANSMISSION = (
    [
        Message(
            MessageType.MODEL,
            b"\x45\x87\x12\xAC",
            FlowControlFlags.TRANSMISSION,
            Flags(
                {
                    FlagName.SUCCESS: True,
                    FlagName.FIRST: True,
                    FlagName.IS_HOST_MESSAGE: True,
                    FlagName.HAS_PAYLOAD: True,
                }
            ),
        ),
        Message(
            MessageType.MODEL,
            b"\x87\x12\xFE",
            FlowControlFlags.TRANSMISSION,
            Flags(
                {
                    FlagName.HAS_PAYLOAD: True,
                }
            ),
        ),
        Message(
            MessageType.MODEL,
            b"",
            FlowControlFlags.TRANSMISSION,
            Flags(
                {
                    FlagName.HAS_PAYLOAD: True,
                }
            ),
        ),
        Message(
            MessageType.MODEL,
            b"\x00\x00\xFF\xFE",
            FlowControlFlags.TRANSMISSION,
            Flags(
                {
                    FlagName.HAS_PAYLOAD: True,
                }
            ),
        ),
        Message(
            MessageType.MODEL,
            b"\xC5\x7D\xAC",
            FlowControlFlags.TRANSMISSION,
            Flags(
                {
                    FlagName.LAST: True,
                    FlagName.HAS_PAYLOAD: True,
                }
            ),
        ),
    ],
    MessageType.MODEL,
    b"\x45\x87\x12\xAC\x87\x12\xFE\x00\x00\xFF\xFE\xC5\x7D\xAC",
    [TransmissionFlag.SUCCESS, TransmissionFlag.IS_HOST_MESSAGE],
)

# Example transmission of type IO_SPEC (only 1 message)
# with validation values (message type, expected payload and
# expected transmission flags).
IO_SPEC_TRANSMISSION_SINGLE_MESSAGE = (
    [
        Message(
            MessageType.IO_SPEC,
            b"\x45\x87\x12\xAC\x12\xFE\x00\x00\xFF",
            FlowControlFlags.TRANSMISSION,
            Flags(
                {
                    FlagName.FAIL: True,
                    FlagName.FIRST: True,
                    FlagName.LAST: True,
                    FlagName.HAS_PAYLOAD: False,
                    FlagName.SPEC_FLAG_1: True,
                }
            ),
        ),
    ],
    MessageType.IO_SPEC,
    b"\x45\x87\x12\xAC\x12\xFE\x00\x00\xFF",
    [TransmissionFlag.FAIL, TransmissionFlag.SERIALIZED],
)

# Example transmission of type IO_SPEC (only 1 message with no payload)
# with validation values (message type, expected payload and
# expected transmission flags).
IO_SPEC_TRANSMISSION_NO_PAYLOAD = (
    [
        Message(
            MessageType.IO_SPEC,
            b"",
            FlowControlFlags.TRANSMISSION,
            Flags(
                {
                    FlagName.FAIL: True,
                    FlagName.FIRST: True,
                    FlagName.LAST: True,
                    FlagName.HAS_PAYLOAD: False,
                    FlagName.SPEC_FLAG_1: False,
                }
            ),
        ),
    ],
    MessageType.IO_SPEC,
    b"",
    [TransmissionFlag.FAIL],
)


def signal_callback_method_path(tested_class: str) -> str:
    """
    Function returns path to the 'signal_callback' method in a given inheriting
    class of ProtocolEvent. It is used to easily create mocks of that method
    for different classes.

    Parameters
    ----------
    tested_class: str
        Class name in the kenning_protocol.py file.

    Returns
    -------
    str
        Python path to the 'signal_callback' method in the given class.
    """
    return f"kenning.protocols.kenning_protocol.{tested_class}.signal_callback"


def assert_signal_callback_mock_incoming_transmission_success(
    test_mock: Callable[Tuple[bool, ProtocolEvent], None],
    desired_call_count: int,
    valid_message_type: MessageType,
    valid_payload: bytes,
    valid_flags: List[TransmissionFlag],
):
    """
    Asserts, that the 'signal_callback' method mock in ProtocolEvent was called
    as it should be after a successful transmission.

    Parameters
    ----------
    test_mock: Callable[Tuple[bool, ProtocolEvent], None]
        The mock to assert.
    desired_call_count: int
        Integer informing how many times the mocked method was expected to be
        called.
    valid_message_type: MessageType
        Expected message type.
    valid_payload: bytes
        Expected payload.
    valid_flags: List[TransmissionFlag]
        Expected transmission flags.
    """
    assert desired_call_count == test_mock.call_count
    if desired_call_count != 0:
        is_successful, event = test_mock.call_args.args
        assert is_successful
        assert type(event) is IncomingTransmission
        assert event.is_completed()
        assert event.has_succeeded()
        (
            received_message_type,
            received_payload,
            received_flags,
        ) = event.get_contents()
        assert valid_message_type == received_message_type
        assert valid_payload == received_payload
        assert valid_flags == received_flags


def assert_signal_callback_mock_incoming_transmission_failure(
    test_mock: Callable[Tuple[bool, ProtocolEvent], None],
    desired_call_count: int,
):
    """
    Asserts, that the 'signal_callback' method mock in ProtocolEvent was called
    as it should be after a failed transmission.

    Parameters
    ----------
    test_mock: Callable[Tuple[bool, ProtocolEvent], None]
        The mock to assert.
    desired_call_count: int
        Integer informing how many times the mocked method was expected to be
        called.
    """
    assert desired_call_count == test_mock.call_count
    if desired_call_count != 0:
        is_successful, event = test_mock.call_args.args
        assert not is_successful
        assert type(event) is IncomingTransmission
        assert event.is_completed()
        assert not event.has_succeeded()


def assert_signal_callback_mock_incoming_request_success(
    test_mock: Callable[Tuple[bool, ProtocolEvent], None],
    desired_call_count: int,
):
    """
    Asserts, that the 'signal_callback' method mock in ProtocolEvent was called
    as it should be after a successfully received request.

    Parameters
    ----------
    test_mock: Callable[Tuple[bool, ProtocolEvent], None]
        The mock to assert.
    desired_call_count: int
        Integer informing how many times the mocked method was expected to be
        called.
    """
    assert desired_call_count == test_mock.call_count
    if desired_call_count != 0:
        is_successful, event = test_mock.call_args.args
        assert is_successful
        assert type(event) is IncomingRequest
        assert event.is_completed()
        assert event.has_succeeded()


def assert_signal_callback_mock_request_failure(
    test_mock: Callable[Tuple[bool, ProtocolEvent], None],
    desired_call_count: int,
):
    """
    Asserts, that the 'signal_callback' method mock in ProtocolEvent was called
    as it should be after a failure in sending a request.

    Parameters
    ----------
    test_mock: Callable[Tuple[bool, ProtocolEvent], None]
        The mock to assert.
    desired_call_count: int
        Integer informing how many times the mocked method was expected to be
        called.
    """
    assert desired_call_count == test_mock.call_count
    if desired_call_count != 0:
        is_successful, event = test_mock.call_args.args
        assert not is_successful
        assert type(event) is OutgoingRequest
        assert event.is_completed()
        assert not event.has_succeeded()


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


class TestProtocolEvent:
    @pytest.mark.parametrize(
        "event_success",
        [
            True,
            False,
        ],
    )
    def test_blocking_mode(
        self,
        protocol_event: ProtocolEvent,
        message_type: MessageType,
        event_success: bool,
    ):
        example_object = IncomingRequest(message_type)
        protocol_event.start_blocking()

        def test_thread():
            protocol_event.signal_callback(event_success, example_object)

        thread = Thread(target=test_thread)
        thread.start()
        is_successful, returned_object = protocol_event.wait()
        assert event_success == is_successful
        assert example_object == returned_object
        thread.join()

    @pytest.mark.parametrize(
        "event_success",
        [
            True,
            False,
        ],
    )
    def test_non_blocking_mode(
        self,
        protocol_event: ProtocolEvent,
        message_type: MessageType,
        event_success: bool,
    ):
        example_object = IncomingRequest(message_type)
        callback_finished_event = Event()

        def test_success_callback(protocol_event: ProtocolEvent):
            assert example_object == protocol_event
            assert event_success
            callback_finished_event.set()

        def test_deny_callback(protocol_event: ProtocolEvent):
            assert example_object == protocol_event
            assert not event_success
            callback_finished_event.set()

        protocol_event.start(test_success_callback, test_deny_callback)
        protocol_event.signal_callback(event_success, example_object)
        callback_finished_event.wait()


class TestOutgoingTransmission:
    @pytest.mark.parametrize(
        "message_type, payload_size, flags",
        [
            (
                MessageType.MODEL,
                1000,
                [TransmissionFlag.IS_HOST_MESSAGE, TransmissionFlag.SUCCESS],
            ),
            (
                MessageType.OUTPUT,
                1000,
                [
                    TransmissionFlag.IS_HOST_MESSAGE,
                    TransmissionFlag.SERIALIZED,
                ],
            ),
            (
                MessageType.IO_SPEC,
                1000,
                [
                    TransmissionFlag.IS_HOST_MESSAGE,
                    TransmissionFlag.SERIALIZED,
                ],
            ),
            (
                MessageType.MODEL,
                5968,
                [TransmissionFlag.IS_HOST_MESSAGE, TransmissionFlag.SUCCESS],
            ),
            (
                MessageType.IO_SPEC,
                10000000,
                [
                    TransmissionFlag.IS_HOST_MESSAGE,
                    TransmissionFlag.SERIALIZED,
                ],
            ),
        ],
    )
    def test_full_lifecycle(
        self,
        message_type: MessageType,
        payload_size: int,
        flags: List[TransmissionFlag],
    ):
        # Generating payload
        payload = bytes([i % 256 for i in range(payload_size)])
        transmission = OutgoingTransmission(message_type, payload, flags)
        with patch(
            signal_callback_method_path("OutgoingTransmission")
        ) as signal_callback_mock:
            transmission.start_blocking()
            message_number = ceil(payload_size / MAX_MESSAGE_PAYLOAD_SIZE)
            # Checking all outgoing messages
            for i in range(message_number):
                assert transmission.has_messages_to_send()
                assert not transmission.accepts_messages()
                message = transmission.get_next_message()
                msg_flags = message.flags
                assert msg_flags.has_payload
                # If it's the first message we test user facing flags and the
                # FIRST flag.
                if i == 0:
                    assert msg_flags.first
                    # Unless there's only 1 message, first message should not
                    # be the last.
                    if message_number != 1:
                        assert not msg_flags.last
                    for key, value in Transmission.FLAG_BINDINGS.items():
                        if key in flags and key.for_type(message_type):
                            assert getattr(msg_flags, str(value))
                        else:
                            assert not getattr(msg_flags, str(value))
                elif i == (message_number - 1):
                    assert msg_flags.last
                    if message_number != 1:
                        assert not msg_flags.first
                else:
                    assert not msg_flags.first
                    assert not msg_flags.last
                # Checking other message properties
                assert (
                    payload[
                        MAX_MESSAGE_PAYLOAD_SIZE * i : MAX_MESSAGE_PAYLOAD_SIZE
                        * (i + 1)
                    ]
                    == message.payload
                )
                assert (
                    FlowControlFlags.TRANSMISSION == message.flow_control_flags
                )
            # Checking if the transmission in the final state
            assert not transmission.has_messages_to_send()
            assert not transmission.accepts_messages()
            assert transmission.has_succeeded()
            assert 1 == signal_callback_mock.call_count
            is_successful, event = signal_callback_mock.call_args.args
            assert is_successful
            assert type(event) is OutgoingTransmission


class TestIncomingTransmission:
    @staticmethod
    def _assert_state_receiving(transmission: IncomingTransmission):
        assert IncomingTransmission.State.RECEIVING == transmission.state
        assert transmission.accepts_messages()
        assert not transmission.has_messages_to_send()
        assert not transmission.has_succeeded()
        assert not transmission.is_completed()

    @staticmethod
    def _assert_state_confirmed(transmission: IncomingTransmission):
        assert IncomingTransmission.State.CONFIRMED == transmission.state
        assert not transmission.accepts_messages()
        assert not transmission.has_messages_to_send()
        assert transmission.has_succeeded()
        assert transmission.is_completed()

    @staticmethod
    def _assert_state_denied(transmission: IncomingTransmission):
        assert IncomingTransmission.State.DENIED == transmission.state
        assert not transmission.accepts_messages()
        assert not transmission.has_messages_to_send()
        assert not transmission.has_succeeded()
        assert transmission.is_completed()

    @staticmethod
    def _get_receiving_object(
        message_type: MessageType,
        timeout: float = -1.0,
        timestamp: float = time.perf_counter(),
    ):
        transmission = IncomingTransmission(message_type)
        transmission.state = IncomingTransmission.State.RECEIVING
        transmission.timeout = timeout
        transmission.flags = Flags()
        transmission.run_callback_on_thread = False
        transmission.completed_event = EventWithArgs()
        transmission.last_message_timestamp = timestamp
        return transmission

    @pytest.mark.parametrize(
        "messages, message_type, payload, flags",
        [
            MODEL_TRANSMISSION,
            IO_SPEC_TRANSMISSION_SINGLE_MESSAGE,
            IO_SPEC_TRANSMISSION_NO_PAYLOAD,
        ],
    )
    def test_full_lifecycle_success(
        self,
        messages: List[Message],
        message_type: MessageType,
        payload: bytes,
        flags: List[TransmissionFlag],
    ):
        transmission = IncomingTransmission(message_type)
        with patch(
            signal_callback_method_path("IncomingTransmission")
        ) as signal_callback_mock:
            transmission.start_blocking()
            for message in messages:
                self._assert_state_receiving(transmission)
                transmission.receive_message(message)
            self._assert_state_confirmed(transmission)
            assert_signal_callback_mock_incoming_transmission_success(
                signal_callback_mock, 1, message_type, payload, flags
            )

    def test_full_lifecycle_timeout(self):
        timeout = 0.001
        transmission = IncomingTransmission(MessageType.MODEL, timeout)
        with patch(
            signal_callback_method_path("IncomingTransmission")
        ) as signal_callback_mock:
            transmission.start_blocking()
            transmission.receive_message(MODEL_TRANSMISSION[0][0])
            timestamp = transmission.last_message_timestamp
            with patch(
                "time.perf_counter", return_value=timestamp + (timeout / 2)
            ):
                transmission.refresh()
            self._assert_state_receiving(transmission)
            with patch(
                "time.perf_counter", return_value=timestamp + 2 * timeout
            ):
                transmission.refresh()
            assert transmission.is_completed()
            assert not transmission.has_succeeded()
            assert transmission.State.DENIED == transmission.state
            assert_signal_callback_mock_incoming_transmission_failure(
                signal_callback_mock, 1
            )

    @pytest.mark.parametrize(
        "first_flag_set,last_flag_set, state_checker",
        [
            (False, True, _assert_state_confirmed),
            (False, False, _assert_state_receiving),
            (True, False, _assert_state_receiving),
            (True, True, _assert_state_confirmed),
        ],
    )
    def test_receive_message(
        self,
        message_type: MessageType,
        first_flag_set: bool,
        last_flag_set: bool,
        state_checker: Callable[ProtocolEvent, None],
    ):
        transmission = self._get_receiving_object(message_type)
        payload = b"\x0B\xDC\x2E"
        with patch(
            signal_callback_method_path("IncomingTransmission")
        ) as signal_callback_mock:
            transmission.receive_message(
                Message(
                    message_type,
                    payload,
                    FlowControlFlags.TRANSMISSION,
                    Flags(
                        {
                            FlagName.LAST: last_flag_set,
                            FlagName.FIRST: first_flag_set,
                            FlagName.FAIL: True,
                        }
                    ),
                )
            )
            state_checker(transmission)
            # Callback should trigger after receiving the last message
            assert_signal_callback_mock_incoming_transmission_success(
                signal_callback_mock,
                1 if last_flag_set else 0,
                message_type,
                payload,
                # Flag from the message should be passed to the callback,
                # only if it was the first message (because user-facing
                # flags are sent in the first message)
                [TransmissionFlag.FAIL] if first_flag_set else [],
            )

    @pytest.mark.parametrize(
        "timeout,time_passed,denied",
        [
            (
                5.0,
                4.3,
                False,
            ),
            (
                2.3,
                5.0,
                True,
            ),
            (
                -1.0,
                45488485565465475766576578.0,
                False,
            ),
        ],
    )
    def test_refresh(
        self,
        message_type: MessageType,
        timeout: float,
        time_passed: float,
        denied: bool,
    ):
        transmission = self._get_receiving_object(message_type, timeout, 0)
        with patch(
            signal_callback_method_path("Transmission")
        ) as signal_callback_mock:
            with patch("time.perf_counter", return_value=time_passed):
                transmission.refresh()
            if denied:
                self._assert_state_denied(transmission)
            else:
                self._assert_state_receiving(transmission)
            assert_signal_callback_mock_incoming_transmission_failure(
                signal_callback_mock, 1 if denied else 0
            )


class TestOutgoingRequest:
    # Helper functions for verifying if the request is in the correct state:

    @staticmethod
    def _assert_state_pending(request: OutgoingRequest):
        assert not request.accepts_messages()
        with pytest.raises(ValueError):
            request.receive_message(MODEL_TRANSMISSION[0][0])
        assert request.has_messages_to_send()
        assert OutgoingRequest.State.PENDING == request.state
        assert not request.is_completed()
        assert not request.has_succeeded()

    @staticmethod
    def _assert_state_sent(request: OutgoingRequest):
        assert OutgoingRequest.State.SENT == request.state
        assert request.request_sent_timestamp is not None
        assert request.accepts_messages()
        assert not request.has_messages_to_send()
        assert not request.is_completed()
        assert not request.has_succeeded()

    @staticmethod
    def _assert_state_accepted(request: OutgoingRequest):
        assert OutgoingRequest.State.ACCEPTED == request.state
        assert request.incoming_transmission is not None
        assert request.accepts_messages()
        assert not request.has_messages_to_send()
        assert not request.is_completed()
        assert not request.has_succeeded()

    @staticmethod
    def _assert_state_confirmed(request: OutgoingRequest):
        assert OutgoingRequest.State.CONFIRMED == request.state
        assert not request.accepts_messages()
        assert not request.has_messages_to_send()
        assert request.is_completed()
        assert request.has_succeeded()

    @staticmethod
    def _assert_state_denied(request: OutgoingRequest):
        assert OutgoingRequest.State.DENIED == request.state
        assert not request.accepts_messages()
        assert not request.has_messages_to_send()
        assert request.is_completed()
        assert not request.has_succeeded()

    # Helper function for checking if the request message
    # generated by the tested class is correct
    @staticmethod
    def _check_request_message(message: Message, message_type: MessageType):
        assert FlowControlFlags.REQUEST == message.flow_control_flags
        assert b"" == message.payload
        assert message.flags.first
        assert message.flags.last
        assert not message.flags.has_payload
        assert not message.flags.success
        assert not message.flags.fail
        assert message_type == message.message_type

    # Helper functions for getting object in a specific state

    @staticmethod
    def _get_sent_object(
        message_type: MessageType,
        retries: int = 0,
        timeout: float = -1.0,
        timestamp: float = time.perf_counter(),
    ):
        request = OutgoingRequest(message_type)
        request.state = OutgoingRequest.State.SENT
        request.timeout = timeout
        request.retry = retries
        request.run_callback_on_thread = False
        request.completed_event = EventWithArgs()
        request.request_sent_timestamp = timestamp
        return request

    @staticmethod
    def _get_accepted_object(
        message_type: MessageType,
        retries: int = 0,
        timeout: float = -1.0,
        timestamp: float = time.perf_counter(),
    ):
        request = OutgoingRequest(message_type)
        request.state = OutgoingRequest.State.ACCEPTED
        request.retry = retries
        request.incoming_transmission = IncomingTransmission(message_type)
        request.incoming_transmission.state = (
            IncomingTransmission.State.RECEIVING
        )
        request.incoming_transmission.run_callback_on_thread = False
        request.incoming_transmission.completed_event = EventWithArgs()
        request.incoming_transmission.last_message_timestamp = timestamp
        request.incoming_transmission.timeout = timeout
        request.run_callback_on_thread = False
        request.completed_event = EventWithArgs()
        request.request_sent_timestamp = None
        return request

    @pytest.mark.parametrize(
        "messages, message_type, payload, flags",
        [
            MODEL_TRANSMISSION,
            IO_SPEC_TRANSMISSION_SINGLE_MESSAGE,
            IO_SPEC_TRANSMISSION_NO_PAYLOAD,
        ],
    )
    def test_full_lifecycle_success(
        self,
        messages: List[Message],
        message_type: MessageType,
        payload: bytes,
        flags: List[TransmissionFlag],
    ):
        request = OutgoingRequest(message_type)
        with patch(
            signal_callback_method_path("OutgoingRequest")
        ) as signal_callback_mock:
            request.start_blocking()
            # State before sending request
            self._assert_state_pending(request)
            request_message = request.get_next_message()
            # OutgoingRequest message assertions
            self._check_request_message(request_message, message_type)
            # State after sending request
            self._assert_state_sent(request)
            for i in range(len(messages)):
                request.receive_message(messages[i])
                # If it's not the last message, after receiving the message
                # state should be changed to ACCEPTED
                if i != (len(messages) - 1):
                    self._assert_state_accepted(request)
            self._assert_state_confirmed(request)
            assert_signal_callback_mock_incoming_transmission_success(
                signal_callback_mock, 1, message_type, payload, flags
            )

    @pytest.mark.parametrize(
        "retries",
        [
            0,
            1,
            67,
        ],
    )
    def test_full_lifecycle_deny(self, retries: int):
        request = OutgoingRequest(message_type, 3, retries)
        with patch(
            signal_callback_method_path("OutgoingRequest")
        ) as signal_callback_mock:
            request.start_blocking()
            self._assert_state_pending(request)
            assert retries == request.retry
            request_message = request.get_next_message()
            self._check_request_message(request_message, message_type)
            self._assert_state_sent(request)
            deny_message = Message(
                message_type,
                b"",
                FlowControlFlags.ACKNOWLEDGE,
                Flags(
                    {
                        FlagName.FAIL: True,
                        FlagName.FIRST: True,
                        FlagName.LAST: True,
                        FlagName.HAS_PAYLOAD: False,
                    }
                ),
            )
            for i in range(retries):
                assert (retries - i) == request.retry
                request.receive_message(deny_message)
                self._assert_state_pending(request)
                request_message = request.get_next_message()
                self._check_request_message(request_message, message_type)
            request.receive_message(deny_message)
            self._assert_state_denied(request)
            assert_signal_callback_mock_request_failure(
                signal_callback_mock, 1
            )

    @pytest.mark.parametrize(
        "retries,timeout",
        [
            (0, 0.001),
            (7, 0.002),
            (15, 0.003),
        ],
    )
    def test_full_lifecycle_timeout(
        self, retries: int, timeout: float, message_type: MessageType
    ):
        request = OutgoingRequest(message_type, timeout, retries)
        with patch(
            signal_callback_method_path("OutgoingRequest")
        ) as signal_callback_mock:
            request.start_blocking()
            self._assert_state_pending(request)
            assert retries == request.retry
            request_message = request.get_next_message()
            self._check_request_message(request_message, message_type)
            self._assert_state_sent(request)
            for i in range(retries):
                assert (retries - i) == request.retry
                timestamp = request.request_sent_timestamp
                with patch(
                    "time.perf_counter", return_value=timestamp + (timeout / 2)
                ):
                    request.refresh()
                self._assert_state_sent(request)
                with patch(
                    "time.perf_counter", return_value=timestamp + 2 * timeout
                ):
                    request.refresh()
                self._assert_state_pending(request)
                request_message = request.get_next_message()
                self._check_request_message(request_message, message_type)
            with patch(
                "time.perf_counter",
                return_value=request.request_sent_timestamp + 2 * timeout,
            ):
                request.refresh()
            self._assert_state_denied(request)
            assert_signal_callback_mock_request_failure(
                signal_callback_mock, 1
            )

    @pytest.mark.parametrize(
        "retries,timeout",
        [
            (0, 0.001),
            (7, 0.002),
            (15, 0.003),
        ],
    )
    def test_full_lifecycle_transmission_timeout(
        self, retries: int, timeout: float
    ):
        message_type = MessageType.MODEL
        request = OutgoingRequest(message_type, timeout, retries)
        with patch(
            signal_callback_method_path("OutgoingRequest")
        ) as signal_callback_mock:
            request.start_blocking()
            self._assert_state_pending(request)
            assert retries == request.retry
            request_message = request.get_next_message()
            self._check_request_message(request_message, message_type)
            self._assert_state_sent(request)
            for i in range(retries):
                assert (retries - i) == request.retry
                request.receive_message(MODEL_TRANSMISSION[0][0])
                timestamp = (
                    request.incoming_transmission.last_message_timestamp
                )
                self._assert_state_accepted(request)
                with patch(
                    "time.perf_counter", return_value=timestamp + timeout / 2
                ):
                    request.refresh()
                self._assert_state_accepted(request)
                with patch(
                    "time.perf_counter", return_value=timestamp + 2 * timeout
                ):
                    request.refresh()
                self._assert_state_pending(request)
                request_message = request.get_next_message()
                self._check_request_message(request_message, message_type)
            request.receive_message(MODEL_TRANSMISSION[0][0])
            with patch(
                "time.perf_counter",
                return_value=request.incoming_transmission.last_message_timestamp
                + 2 * timeout,
            ):
                request.refresh()
            self._assert_state_denied(request)
            assert_signal_callback_mock_request_failure(
                signal_callback_mock, 1
            )

    # Acknowledgment without the FAIL flag set while in SENT state should be
    # discarded without changing state of the object (as should any message
    # other than negative acknowledgment or transmission)
    @pytest.mark.parametrize(
        "fail_flag_set,retries, state_checker",
        [
            (False, 0, _assert_state_sent),
            (False, 1, _assert_state_sent),
            (False, 15, _assert_state_sent),
            (True, 0, _assert_state_denied),
            (True, 1, _assert_state_pending),
            (True, 7, _assert_state_pending),
        ],
    )
    def test_receive_message_acknowledge_sent(
        self,
        message_type: MessageType,
        fail_flag_set: bool,
        retries: int,
        state_checker: Callable[ProtocolEvent, None],
    ):
        request = self._get_sent_object(message_type, retries)
        request.receive_message(
            Message(
                message_type,
                None,
                FlowControlFlags.ACKNOWLEDGE,
                Flags(
                    {
                        FlagName.FAIL: fail_flag_set,
                    }
                ),
            )
        )
        state_checker(request)

    @pytest.mark.parametrize(
        "outgoing_request,last_flag_set,state_checker",
        [
            (
                _get_sent_object(DEFAULT_MESSAGE_TYPE),
                False,
                _assert_state_accepted,
            ),
            (
                _get_sent_object(DEFAULT_MESSAGE_TYPE),
                True,
                _assert_state_confirmed,
            ),
            (
                _get_accepted_object(DEFAULT_MESSAGE_TYPE),
                False,
                _assert_state_accepted,
            ),
            (
                _get_accepted_object(DEFAULT_MESSAGE_TYPE),
                True,
                _assert_state_confirmed,
            ),
        ],
    )
    def test_receive_message_transmission(
        self,
        message_type: MessageType,
        outgoing_request: OutgoingRequest,
        last_flag_set: bool,
        state_checker: Callable[ProtocolEvent, None],
    ):
        with patch(
            signal_callback_method_path("OutgoingRequest")
        ) as signal_callback_mock:
            payload = b"\x32\xF4\xCE"
            outgoing_request.receive_message(
                Message(
                    message_type,
                    payload,
                    FlowControlFlags.TRANSMISSION,
                    Flags(
                        {
                            FlagName.FIRST: True,
                            FlagName.LAST: last_flag_set,
                        }
                    ),
                )
            )
            state_checker(outgoing_request)
            if last_flag_set:
                assert_signal_callback_mock_incoming_transmission_success(
                    signal_callback_mock, 1, message_type, payload, []
                )
            else:
                assert 0 == signal_callback_mock.call_count

    def test_get_next_message_pending(self, message_type: MessageType):
        request = OutgoingRequest(message_type)
        request.state = OutgoingRequest.State.PENDING
        request_message = Message(
            message_type, None, FlowControlFlags.REQUEST, Flags()
        )
        request.request_message = request_message
        assert request_message == request.get_next_message()

    def test_init(self, message_type: MessageType):
        request = OutgoingRequest(message_type)
        self._check_request_message(request.request_message, message_type)

    @pytest.mark.parametrize(
        "object_generator,timeout,retries,time_passed,retries_left,state_checker",
        [
            (
                _get_sent_object,
                5.0,
                2,
                4.3,
                2,
                _assert_state_sent,
            ),
            (
                _get_sent_object,
                -1.0,
                2,
                5843857435743597543574357435747,
                2,
                _assert_state_sent,
            ),
            (
                _get_sent_object,
                2.3,
                0,
                1.0,
                0,
                _assert_state_sent,
            ),
            (
                _get_sent_object,
                5.0,
                2,
                7.3,
                1,
                _assert_state_pending,
            ),
            (
                _get_sent_object,
                2.3,
                0,
                5.0,
                None,
                _assert_state_denied,
            ),
            (
                _get_accepted_object,
                3.1,
                1,
                4.1,
                0,
                _assert_state_pending,
            ),
            (
                _get_accepted_object,
                1.5,
                0,
                2.0,
                None,
                _assert_state_denied,
            ),
            (
                _get_accepted_object,
                3.1,
                9,
                2.1,
                9,
                _assert_state_accepted,
            ),
            (
                _get_accepted_object,
                1.5,
                0,
                1.0,
                0,
                _assert_state_accepted,
            ),
            (
                _get_accepted_object,
                -1.0,
                9,
                25656.1,
                9,
                _assert_state_accepted,
            ),
        ],
    )
    def test_refresh(
        self,
        message_type: MessageType,
        object_generator: Callable[
            Tuple[MessageType, int, float, float], OutgoingRequest
        ],
        timeout: float,
        retries: int,
        time_passed: float,
        retries_left: int,
        state_checker: Callable[ProtocolEvent, None],
    ):
        request = object_generator(message_type, retries, timeout, 0)
        with patch(
            signal_callback_method_path("OutgoingRequest")
        ) as signal_callback_mock:
            with patch("time.perf_counter", return_value=time_passed):
                request.refresh()
            state_checker(request)
            if retries_left is not None:
                assert retries_left == request.retry
            state_denied = time_passed > timeout and retries == 0
            assert_signal_callback_mock_request_failure(
                signal_callback_mock, 1 if state_denied else 0
            )


class TestListen:
    # Helper functions for verifying if the request is in the correct state:

    @staticmethod
    def _assert_state_listening(listen: Listen):
        assert listen.State.LISTENING == listen.state
        assert listen.accepts_messages()
        assert not listen.has_messages_to_send()
        assert not listen.is_completed()
        assert not listen.has_succeeded()

    @staticmethod
    def _assert_state_receiving(listen: Listen):
        assert not listen.is_completed()
        assert not listen.has_succeeded()
        assert listen.State.RECEIVING_TRANSMISSION == listen.state
        assert listen.incoming_transmission is not None

    @staticmethod
    def _assert_state_confirmed(listen: Listen):
        assert listen.is_completed()
        assert listen.has_succeeded()
        assert not listen.accepts_messages()
        assert not listen.has_messages_to_send()
        assert listen.State.CONFIRMED == listen.state

    # Helper functions for getting object in a specific state

    @staticmethod
    def _get_receiving_object(
        message_type: MessageType,
        limit: Optional[int] = 1,
        timeout: float = 0.001,
    ) -> Listen:
        listen = Listen(message_type)
        listen.state = listen.State.RECEIVING_TRANSMISSION
        listen.incoming_transmission = IncomingTransmission(message_type)
        listen.incoming_transmission.state = (
            IncomingTransmission.State.RECEIVING
        )
        listen.incoming_transmission.flags = Flags()
        listen.incoming_transmission.start_blocking()
        listen.incoming_transmission.timeout = timeout
        listen.timeout = timeout
        listen.limit = limit
        return listen

    @staticmethod
    def _get_listening_object(
        message_type: MessageType, limit: Optional[int] = 1
    ) -> Listen:
        listen = Listen(message_type)
        listen.state = listen.State.LISTENING
        listen.limit = limit
        return listen

    @pytest.mark.parametrize(
        "transmission,limit",
        [
            (MODEL_TRANSMISSION, 1),
            (IO_SPEC_TRANSMISSION_SINGLE_MESSAGE, 5),
            (IO_SPEC_TRANSMISSION_NO_PAYLOAD, 15),
        ],
    )
    def test_full_lifecycle_transmission_success(
        self,
        transmission: Tuple[
            List[Message], MessageType, bytes, List[TransmissionFlag]
        ],
        limit: int,
    ):
        messages, message_type, payload, flags = transmission
        listen = Listen(message_type, limit)
        with patch(
            signal_callback_method_path("Listen")
        ) as signal_callback_mock:

            def do_nothing_callback(event: ProtocolEvent):
                pass

            listen.start(do_nothing_callback, do_nothing_callback)
            for i in range(limit):
                self._assert_state_listening(listen)
                for j in range(len(messages)):
                    listen.receive_message(messages[j])
                    if j != len(messages) - 1:
                        self._assert_state_receiving(listen)
                assert_signal_callback_mock_incoming_transmission_success(
                    signal_callback_mock, i + 1, message_type, payload, flags
                )
            self._assert_state_confirmed(listen)

    def test_full_lifecycle_transmission_timeout(self):
        timeout = 0.001
        listen = Listen(MessageType.MODEL, 2, timeout)
        with patch(
            signal_callback_method_path("Listen")
        ) as signal_callback_mock:

            def do_nothing_callback(event: ProtocolEvent):
                pass

            listen.start(do_nothing_callback, do_nothing_callback)
            self._assert_state_listening(listen)
            # Limit is set to 2, so after first failed transmission the object
            # should go back to state 'LISTENING'
            listen.receive_message(MODEL_TRANSMISSION[0][0])
            self._assert_state_receiving(listen)
            timestamp = listen.incoming_transmission.last_message_timestamp
            with patch(
                "time.perf_counter", return_value=timestamp + (timeout / 2)
            ):
                listen.refresh()
            self._assert_state_receiving(listen)
            with patch(
                "time.perf_counter", return_value=timestamp + 2 * timeout
            ):
                listen.refresh()
            self._assert_state_listening(listen)
            assert_signal_callback_mock_incoming_transmission_failure(
                signal_callback_mock, 1
            )
            # Second failed transmission - object IncomingTransmission passed
            # to the callback should be in state 'DENIED', but the listening
            # object itself in state 'CONFIRMED'
            listen.receive_message(MODEL_TRANSMISSION[0][0])
            timestamp = listen.incoming_transmission.last_message_timestamp
            with patch(
                "time.perf_counter", return_value=timestamp + (timeout / 2)
            ):
                listen.refresh()
            self._assert_state_receiving(listen)
            with patch(
                "time.perf_counter", return_value=timestamp + 2 * timeout
            ):
                listen.refresh()
            self._assert_state_confirmed(listen)
            assert_signal_callback_mock_incoming_transmission_failure(
                signal_callback_mock, 2
            )

    @pytest.mark.parametrize(
        "limit",
        [
            1,
            5,
            15,
        ],
    )
    def test_full_lifecycle_request(
        self, message_type: MessageType, limit: int
    ):
        listen = Listen(message_type, limit)
        with patch(
            signal_callback_method_path("Listen")
        ) as signal_callback_mock:

            def do_nothing_callback(event: ProtocolEvent):
                pass

            listen.start(do_nothing_callback, do_nothing_callback)
            for i in range(limit):
                self._assert_state_listening(listen)
                listen.receive_message(
                    Message(
                        message_type,
                        None,
                        FlowControlFlags.REQUEST,
                        Flags(
                            {
                                FlagName.FIRST: True,
                                FlagName.LAST: True,
                            }
                        ),
                    )
                )
                assert_signal_callback_mock_incoming_request_success(
                    signal_callback_mock, i + 1
                )
            self._assert_state_confirmed(listen)

    @pytest.mark.parametrize(
        "limit",
        [
            1,
            5,
            15,
            None,
        ],
    )
    def test_start_blocking(
        self, message_type: MessageType, limit: Optional[int]
    ):
        # In blocking mode limit other than one is not supported
        listen = Listen(message_type, limit)
        listen.start_blocking()
        assert 1 == listen.limit
        assert listen.State.LISTENING == listen.state

    # Listen object in state LISTENING, after receiving a TRANSMISSION
    # message should change state to RECEIVING_TRANSMISSION (unless LAST
    # flag in the message is set, because that would mean it's a 1-message
    # transmission - so the Listen object should receive it and return to
    # LISTENING state).
    def test_receive_message_transmission_listening(
        self, message_type: MessageType
    ):
        listen = Listen(message_type)
        listen.state = listen.State.LISTENING
        transmission_message = Message(
            message_type,
            None,
            FlowControlFlags.TRANSMISSION,
            Flags(
                {
                    FlagName.FIRST: True,
                }
            ),
        )
        listen.receive_message(transmission_message)
        self._assert_state_receiving(listen)
        assert listen.incoming_transmission is not None
        assert (
            IncomingTransmission.State.RECEIVING
            == listen.incoming_transmission.state
        )

    # Listen object in state LISTENING or RECEIVING, after receiving a
    # request, should signal callbacks (passing an IncomingRequest object)
    @pytest.mark.parametrize(
        "listen,state_checker",
        [
            (
                _get_receiving_object(DEFAULT_MESSAGE_TYPE, 5),
                _assert_state_receiving,
            ),
            (
                _get_listening_object(DEFAULT_MESSAGE_TYPE, 5),
                _assert_state_listening,
            ),
        ],
    )
    def test_receive_message_request(
        self,
        message_type: MessageType,
        listen: Listen,
        state_checker: Callable[ProtocolEvent, None],
    ):
        with patch(
            signal_callback_method_path("Listen")
        ) as signal_callback_mock:
            test_limit = listen.limit
            request_message = Message(
                message_type, None, FlowControlFlags.REQUEST, Flags()
            )
            listen.receive_message(request_message)
            state_checker(listen)
            assert test_limit - 1 == listen.limit
            assert_signal_callback_mock_incoming_request_success(
                signal_callback_mock, 1
            )

    def test_receive_message_last_transmission_receiving(
        self, message_type: MessageType
    ):
        test_limit = 5
        listen = self._get_receiving_object(message_type, test_limit)
        with patch(
            signal_callback_method_path("Listen")
        ) as signal_callback_mock:
            payload = b"\xFE\x32\x56"
            listen.receive_message(
                Message(
                    message_type,
                    payload,
                    FlowControlFlags.TRANSMISSION,
                    Flags(
                        {
                            FlagName.LAST: True,
                        }
                    ),
                )
            )
            self._assert_state_listening(listen)
            assert test_limit - 1 == listen.limit
            assert_signal_callback_mock_incoming_transmission_success(
                signal_callback_mock, 1, message_type, payload, []
            )

    @pytest.mark.parametrize(
        "timeout_passed,limit,state_checker,desired_limit",
        [
            (False, 1, _assert_state_receiving, None),
            (True, 1, _assert_state_confirmed, None),
            (True, 2, _assert_state_listening, 1),
            (True, 18, _assert_state_listening, 17),
        ],
    )
    def test_refresh_timeout(
        self,
        message_type: MessageType,
        timeout_passed: bool,
        limit: int,
        state_checker: Callable[ProtocolEvent, None],
        desired_limit: int,
    ):
        timeout = 1.0
        listen = self._get_receiving_object(message_type, limit, timeout)
        # We're tricking the object into thinking the timeout has passed or not
        listen.incoming_transmission.last_message_timestamp = (
            (time.perf_counter() - timeout)
            if timeout_passed
            else time.perf_counter()
        )
        with patch(
            signal_callback_method_path("Listen")
        ) as signal_callback_mock:
            listen.refresh()
            assert_signal_callback_mock_incoming_transmission_failure(
                signal_callback_mock, 1 if timeout_passed else 0
            )

            state_checker(listen)
            if desired_limit is not None:
                assert desired_limit == listen.limit
