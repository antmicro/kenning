# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import multiprocessing
from math import ceil
from multiprocessing.pool import ThreadPool
from threading import Event, Lock, Thread
from typing import Any, Callable, List, Optional, Tuple
from unittest.mock import Mock, patch

import pytest

from kenning.protocols.bytes_based_protocol import TransmissionFlag
from kenning.protocols.kenning_protocol import (
    FLAG_BINDINGS,
    FlowControlFlags,
    IncomingEventType,
    IncomingRequest,
    IncomingTransmission,
    KenningProtocol,
    Listen,
    MessageType,
    OutgoingRequest,
    OutgoingTransmission,
    ProtocolEvent,
)
from kenning.protocols.message import HEADER_SIZE, FlagName, Flags, Message
from kenning.tests.utils.test_serializable import serializables_equal
from kenning.utils.event_with_args import EventWithArgs

DEFAULT_MESSAGE_TYPE = MessageType.DATA


@pytest.fixture
def message_type():
    return DEFAULT_MESSAGE_TYPE


@pytest.fixture
@patch.multiple(KenningProtocol, __abstractmethods__=set())
def protocol():
    protocol = KenningProtocol()
    return protocol


@pytest.fixture
@patch.multiple(ProtocolEvent, __abstractmethods__=set())
def protocol_event(message_type: MessageType, protocol: KenningProtocol):
    protocol_event = ProtocolEvent(message_type, protocol)
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
    valid_message_type: MessageType,
    valid_payload: bytes,
    valid_flags: List[TransmissionFlag],
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
        (
            received_message_type,
            received_payload,
            received_flags,
        ) = event.get_contents()
        assert valid_message_type == received_message_type
        assert valid_payload == received_payload
        assert valid_flags == received_flags


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


class MockKenningProtocol(KenningProtocol):
    """
    Fake KenningProtocol class, that can be passed to ProtocolEvent
    object. It overrides send_messages method, that is called by
    some ProtocolEvent objects, so that it doesn't raise an exception
    trying to actually send messages.
    """

    def __init__(self):
        self.receiver_thread = None
        self.transmitter = None
        self.receiver_running = False

    def send_messages(self, message_type: MessageType, messages):
        pass

    def disconnect(self):
        pass

    def initialize_server(
        self,
        client_connected_callback: Optional[Callable[Any, None]] = None,
        client_disconnected_callback: Optional[Callable[None, None]] = None,
    ):
        pass

    def initialize_client(self):
        pass

    def receive_data(self, timeout: float):
        pass

    def send_data(self, data: bytes):
        pass


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
        protocol: KenningProtocol,
        protocol_event: ProtocolEvent,
        message_type: MessageType,
        event_success: bool,
    ):
        example_object = IncomingRequest(message_type, protocol)
        protocol_event.start_blocking()

        def test_thread():
            protocol_event.signal_callback(event_success, example_object)

        thread = Thread(target=test_thread)
        thread.start()
        is_successful, returned_object = protocol_event.wait(120)
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
        protocol: KenningProtocol,
        protocol_event: ProtocolEvent,
        message_type: MessageType,
        event_success: bool,
    ):
        example_object = IncomingRequest(message_type, protocol)
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
        "message_type, payload_size, flags, max_message_size",
        [
            (
                MessageType.MODEL,
                1000,
                [TransmissionFlag.IS_HOST_MESSAGE, TransmissionFlag.SUCCESS],
                20,
            ),
            (
                MessageType.OUTPUT,
                1000,
                [
                    TransmissionFlag.IS_HOST_MESSAGE,
                    TransmissionFlag.SERIALIZED,
                ],
                28339478,
            ),
            (
                MessageType.IO_SPEC,
                1000,
                [
                    TransmissionFlag.IS_HOST_MESSAGE,
                    TransmissionFlag.SERIALIZED,
                ],
                1000,
            ),
            (
                MessageType.MODEL,
                5968,
                [TransmissionFlag.IS_HOST_MESSAGE, TransmissionFlag.SUCCESS],
                6700,
            ),
            (
                MessageType.IO_SPEC,
                10000000,
                [
                    TransmissionFlag.IS_HOST_MESSAGE,
                    TransmissionFlag.SERIALIZED,
                ],
                13,
            ),
        ],
    )
    def test_full_lifecycle(
        self,
        message_type: MessageType,
        payload_size: int,
        flags: List[TransmissionFlag],
        max_message_size: int,
    ):
        # Generating payload
        payload = bytes([i % 256 for i in range(payload_size)])
        protocol_mock = Mock()
        protocol_mock.max_message_size = max_message_size
        max_message_payload_size = max_message_size - HEADER_SIZE
        transmission = OutgoingTransmission(
            message_type, protocol_mock, payload, flags
        )
        with (
            patch(
                signal_callback_method_path("OutgoingTransmission")
            ) as signal_callback_mock,
            patch.object(
                transmission.protocol, "send_messages"
            ) as send_messages_mock,
        ):
            transmission.start_blocking()
            message_number = ceil(payload_size / max_message_payload_size)
            # Checking all outgoing messages
            assert 1 == send_messages_mock.call_count
            _message_type, messages = send_messages_mock.call_args.args
            assert message_number == len(messages)
            assert message_type == _message_type
            transmission.messages_sent(message_number)
            for i in range(len(messages)):
                message = messages[i]
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
                    for key, value in FLAG_BINDINGS.items():
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
                        max_message_payload_size * i : max_message_payload_size
                        * (i + 1)
                    ]
                    == message.payload
                )
                assert (
                    FlowControlFlags.TRANSMISSION == message.flow_control_flags
                )
            # Checking if the transmission in the final state
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
        assert not transmission.has_succeeded()
        assert not transmission.is_completed()

    @staticmethod
    def _assert_state_confirmed(transmission: IncomingTransmission):
        assert IncomingTransmission.State.CONFIRMED == transmission.state
        assert not transmission.accepts_messages()
        assert transmission.has_succeeded()
        assert transmission.is_completed()

    @staticmethod
    def _assert_state_denied(transmission: IncomingTransmission):
        assert IncomingTransmission.State.DENIED == transmission.state
        assert not transmission.accepts_messages()
        assert not transmission.has_succeeded()
        assert transmission.is_completed()

    @staticmethod
    def _get_receiving_object(
        message_type: MessageType,
    ):
        transmission = IncomingTransmission(message_type, Mock())
        transmission.state = IncomingTransmission.State.RECEIVING
        transmission.flags = Flags()
        transmission.run_callback_on_thread = False
        transmission.completed_event = EventWithArgs()
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
        transmission = IncomingTransmission(message_type, Mock())
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


class TestOutgoingRequest:
    # Helper functions for verifying if the request is in the correct state:

    @staticmethod
    def _assert_state_pending(request: OutgoingRequest):
        assert not request.accepts_messages()
        with pytest.raises(ValueError):
            request.receive_message(MODEL_TRANSMISSION[0][0])
        assert OutgoingRequest.State.PENDING == request.state
        assert not request.is_completed()
        assert not request.has_succeeded()

    @staticmethod
    def _assert_state_sent(request: OutgoingRequest):
        assert OutgoingRequest.State.SENT == request.state
        assert request.accepts_messages()
        assert not request.is_completed()
        assert not request.has_succeeded()

    @staticmethod
    def _assert_state_accepted(request: OutgoingRequest):
        assert OutgoingRequest.State.ACCEPTED == request.state
        assert request.incoming_transmission is not None
        assert request.accepts_messages()
        assert not request.is_completed()
        assert not request.has_succeeded()

    @staticmethod
    def _assert_state_confirmed(request: OutgoingRequest):
        assert OutgoingRequest.State.CONFIRMED == request.state
        assert not request.accepts_messages()
        assert request.is_completed()
        assert request.has_succeeded()

    @staticmethod
    def _assert_state_denied(request: OutgoingRequest):
        assert OutgoingRequest.State.DENIED == request.state
        assert not request.accepts_messages()
        assert request.is_completed()
        assert not request.has_succeeded()

    # Helper functions for checking if the request message
    # generated by the tested class is correct
    @staticmethod
    def _check_request_messages(
        messages: List[Message],
        message_type: MessageType,
        payload: bytes,
        flags: List[TransmissionFlag],
    ):
        rec_payload = b""
        for i in range(len(messages)):
            message = messages[i]
            assert FlowControlFlags.REQUEST == message.flow_control_flags
            rec_payload += message.payload
            if i == 0:
                assert message.flags.first
            else:
                assert not message.flags.first
            if i == len(messages) - 1:
                assert message.flags.last
            else:
                assert not message.flags.last
            if payload == b"":
                assert not message.flags.has_payload
            else:
                assert message.flags.has_payload
            assert not message.flags.success
            assert not message.flags.fail
            for flag in flags:
                if flag.for_type(message_type):
                    assert getattr(message.flags, str(FLAG_BINDINGS[flag]))
            assert message_type == message.message_type
        assert payload == rec_payload

    def _check_request_messages_from_mock(
        self,
        mock: Callable[Tuple[MessageType, List[Message]], None],
        message_type: MessageType,
        payload: bytes = b"",
        flags: List[TransmissionFlag] = [],
    ):
        assert 1 == mock.call_count
        _message_type, messages = mock.call_args.args
        assert message_type == _message_type
        assert 1 == len(messages)
        self._check_request_messages(messages, message_type, payload, flags)
        mock.reset_mock()

    # Helper functions for getting object in a specific state

    @staticmethod
    def _get_new_object(message_type: MessageType):
        mock = Mock()
        mock.max_message_size = 20
        return OutgoingRequest(message_type, mock)

    @staticmethod
    def _get_sent_object(
        message_type: MessageType,
        retries: int = 0,
    ):
        mock = Mock()
        mock.max_message_size = 20
        request = OutgoingRequest(message_type, mock)
        request.state = OutgoingRequest.State.SENT
        request.retry = retries
        request.run_callback_on_thread = False
        request.completed_event = EventWithArgs()
        return request

    @staticmethod
    def _get_accepted_object(
        message_type: MessageType,
        retries: int = 0,
    ):
        mock = Mock()
        mock.max_message_size = 20
        request = OutgoingRequest(message_type, mock)
        request.state = OutgoingRequest.State.ACCEPTED
        request.retry = retries
        request.incoming_transmission = IncomingTransmission(
            message_type, Mock()
        )
        request.incoming_transmission.state = (
            IncomingTransmission.State.RECEIVING
        )
        request.incoming_transmission.run_callback_on_thread = False
        request.incoming_transmission.completed_event = EventWithArgs()
        request.run_callback_on_thread = False
        request.completed_event = EventWithArgs()
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
        request = self._get_new_object(message_type)
        with (
            patch(
                signal_callback_method_path("OutgoingRequest")
            ) as signal_callback_mock,
            patch.object(
                request.protocol, "send_messages"
            ) as send_messages_mock,
        ):
            request.start_blocking()
            # State before sending request
            self._assert_state_pending(request)
            self._check_request_messages_from_mock(
                send_messages_mock, message_type
            )
            request.messages_sent(1)
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
        request = self._get_new_object(message_type)
        request.retry = retries
        assert retries == request.retry
        with (
            patch(
                signal_callback_method_path("OutgoingRequest")
            ) as signal_callback_mock,
            patch.object(
                request.protocol, "send_messages"
            ) as send_messages_mock,
        ):
            request.start_blocking()
            self._assert_state_pending(request)
            self._check_request_messages_from_mock(
                send_messages_mock, message_type
            )
            request.messages_sent(1)

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
                self._check_request_messages_from_mock(
                    send_messages_mock, message_type
                )
                request.messages_sent(1)
            request.receive_message(deny_message)
            self._assert_state_denied(request)
            assert_signal_callback_mock_request_failure(
                signal_callback_mock, 1
            )

    # Acknowledgment without the FAIL flag set while in SENT state should be
    # discarded without changing state of the object (as should any message
    # other than negative acknowledgment or transmission)
    @pytest.mark.parametrize(
        "fail_flag_set,retries, state_checker, request_sent_again",
        [
            (False, 0, _assert_state_sent, False),
            (False, 1, _assert_state_sent, False),
            (False, 15, _assert_state_sent, False),
            (True, 0, _assert_state_denied, False),
            (True, 1, _assert_state_pending, True),
            (True, 7, _assert_state_pending, True),
        ],
    )
    def test_receive_message_acknowledge_sent(
        self,
        message_type: MessageType,
        fail_flag_set: bool,
        retries: int,
        state_checker: Callable[ProtocolEvent, None],
        request_sent_again: bool,
    ):
        request = self._get_sent_object(message_type, retries)
        with patch.object(
            request.protocol, "send_messages"
        ) as send_messages_mock:
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
            if request_sent_again:
                self._check_request_messages_from_mock(
                    send_messages_mock, message_type
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

    def test_messages_sent_pending(self, message_type: MessageType):
        request = self._get_new_object(message_type)
        request.state = OutgoingRequest.State.PENDING
        request.messages_sent(1)
        request.state == OutgoingRequest.State.SENT

    def test_init(self, message_type: MessageType, random_byte_data: bytes):
        mock = Mock()
        mock.max_message_size = 10
        request = OutgoingRequest(
            message_type,
            mock,
            5,
            random_byte_data,
            [TransmissionFlag.IS_HOST_MESSAGE],
        )
        self._check_request_messages(
            request.get_remaining_messages(),
            message_type,
            random_byte_data,
            [TransmissionFlag.IS_HOST_MESSAGE],
        )
        assert 5 == request.retry


class TestListen:
    # Helper functions for verifying if the request is in the correct state:

    @staticmethod
    def _assert_state_listening(listen: Listen):
        assert listen.State.LISTENING == listen.state
        assert listen.accepts_messages()
        assert not listen.is_completed()
        assert not listen.has_succeeded()

    @staticmethod
    def _assert_state_receiving(listen: Listen):
        assert not listen.is_completed()
        assert not listen.has_succeeded()
        assert listen.State.RECEIVING == listen.state
        assert listen.inner_object is not None

    @staticmethod
    def _assert_state_confirmed(listen: Listen):
        assert listen.is_completed()
        assert listen.has_succeeded()
        assert not listen.accepts_messages()
        assert listen.State.CONFIRMED == listen.state

    # Helper functions for getting object in a specific state

    @staticmethod
    def _get_receiving_object(
        message_type: MessageType,
        event_type,
        limit: Optional[int] = 1,
    ) -> Listen:
        listen = Listen(message_type, Mock())
        listen.state = listen.State.RECEIVING
        listen.inner_object = event_type(message_type, Mock())
        listen.inner_object.state = IncomingTransmission.State.RECEIVING
        listen.inner_object.flags = Flags()
        listen.inner_object.start_blocking()
        listen.limit = limit
        return listen

    @staticmethod
    def _get_listening_object(
        message_type: MessageType, limit: Optional[int] = 1
    ) -> Listen:
        listen = Listen(message_type, Mock())
        listen.state = listen.State.LISTENING
        listen.limit = limit
        return listen

    @pytest.mark.parametrize(
        "transmission_or_request,limit,event,mock_asserter",
        [
            (
                MODEL_TRANSMISSION,
                1,
                FlowControlFlags.TRANSMISSION,
                assert_signal_callback_mock_incoming_transmission_success,
            ),
            (
                IO_SPEC_TRANSMISSION_SINGLE_MESSAGE,
                5,
                FlowControlFlags.TRANSMISSION,
                assert_signal_callback_mock_incoming_transmission_success,
            ),
            (
                IO_SPEC_TRANSMISSION_NO_PAYLOAD,
                15,
                FlowControlFlags.TRANSMISSION,
                assert_signal_callback_mock_incoming_transmission_success,
            ),
            (
                MODEL_TRANSMISSION,
                1,
                FlowControlFlags.REQUEST,
                assert_signal_callback_mock_incoming_request_success,
            ),
            (
                IO_SPEC_TRANSMISSION_SINGLE_MESSAGE,
                5,
                FlowControlFlags.REQUEST,
                assert_signal_callback_mock_incoming_request_success,
            ),
            (
                IO_SPEC_TRANSMISSION_NO_PAYLOAD,
                15,
                FlowControlFlags.REQUEST,
                assert_signal_callback_mock_incoming_request_success,
            ),
        ],
    )
    def test_full_lifecycle(
        self,
        transmission_or_request: Tuple[
            List[Message], MessageType, bytes, List[TransmissionFlag]
        ],
        limit: int,
        event: FlowControlFlags,
        mock_asserter: Callable[
            Tuple[
                Callable[Tuple[bool, ProtocolEvent], None],
                int,
                MessageType,
                bytes,
                List[TransmissionFlag],
            ],
            None,
        ],
    ):
        messages, message_type, payload, flags = copy.deepcopy(
            transmission_or_request
        )
        for message in messages:
            message.flow_control_flags = event
        listen = Listen(message_type, Mock(), limit)
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
                mock_asserter(
                    signal_callback_mock, i + 1, message_type, payload, flags
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

    @pytest.mark.parametrize(
        "listen,state_checker,first_flag_set,last_flag_set,expected_limit,flow_control_flags,mock_asserter,signal_callback_called",
        [
            (
                _get_receiving_object(
                    DEFAULT_MESSAGE_TYPE, IncomingRequest, 5
                ),
                _assert_state_listening,
                False,
                True,
                4,
                FlowControlFlags.REQUEST,
                assert_signal_callback_mock_incoming_request_success,
                True,
            ),
            (
                _get_listening_object(DEFAULT_MESSAGE_TYPE, -4),
                _assert_state_listening,
                True,
                True,
                -4,
                FlowControlFlags.REQUEST,
                assert_signal_callback_mock_incoming_request_success,
                True,
            ),
            (
                _get_receiving_object(
                    DEFAULT_MESSAGE_TYPE, IncomingRequest, 0
                ),
                _assert_state_receiving,
                False,
                False,
                0,
                FlowControlFlags.REQUEST,
                assert_signal_callback_mock_incoming_request_success,
                False,
            ),
            (
                _get_listening_object(DEFAULT_MESSAGE_TYPE, 2),
                _assert_state_receiving,
                True,
                False,
                2,
                FlowControlFlags.REQUEST,
                assert_signal_callback_mock_incoming_request_success,
                False,
            ),
            (
                _get_receiving_object(
                    DEFAULT_MESSAGE_TYPE, IncomingTransmission, 14
                ),
                _assert_state_listening,
                False,
                True,
                13,
                FlowControlFlags.TRANSMISSION,
                assert_signal_callback_mock_incoming_transmission_success,
                True,
            ),
            (
                _get_listening_object(DEFAULT_MESSAGE_TYPE, 1),
                _assert_state_confirmed,
                True,
                True,
                0,
                FlowControlFlags.TRANSMISSION,
                assert_signal_callback_mock_incoming_transmission_success,
                True,
            ),
            (
                _get_receiving_object(
                    DEFAULT_MESSAGE_TYPE, IncomingTransmission, 2
                ),
                _assert_state_receiving,
                False,
                False,
                2,
                FlowControlFlags.TRANSMISSION,
                assert_signal_callback_mock_incoming_transmission_success,
                False,
            ),
            (
                _get_listening_object(DEFAULT_MESSAGE_TYPE, 2),
                _assert_state_receiving,
                True,
                False,
                2,
                FlowControlFlags.TRANSMISSION,
                assert_signal_callback_mock_incoming_transmission_success,
                False,
            ),
        ],
    )
    def test_receive_message(
        self,
        message_type: MessageType,
        listen: Listen,
        state_checker: Callable[ProtocolEvent, None],
        first_flag_set: bool,
        last_flag_set: bool,
        expected_limit: int,
        flow_control_flags: FlowControlFlags,
        mock_asserter: Callable[
            Tuple[
                Callable[Tuple[bool, ProtocolEvent], None],
                int,
                MessageType,
                bytes,
                List[TransmissionFlag],
            ],
            None,
        ],
        signal_callback_called: bool,
    ):
        with patch(
            signal_callback_method_path("Listen")
        ) as signal_callback_mock:
            message = Message(
                message_type,
                None,
                flow_control_flags,
                Flags(
                    {
                        FlagName.FIRST: first_flag_set,
                        FlagName.LAST: last_flag_set,
                    }
                ),
            )
            listen.receive_message(message)
            state_checker(listen)
            assert expected_limit == listen.limit
            mock_asserter(
                signal_callback_mock,
                1 if signal_callback_called else 0,
                message_type,
                b"",
                [],
            )


class MockEvent(ProtocolEvent):
    def __init__(
        self,
        message_type: MessageType,
        messages_expected: int,
    ):
        self.state = None
        self.message_type = message_type
        self.rec_messages = []
        self.messages_sent_count = 0
        self.messages_expected = messages_expected
        self.lock = Lock()

    def is_completed(self) -> bool:
        return len(self.rec_messages) == self.messages_expected

    def accepts_messages(self) -> bool:
        return not (len(self.rec_messages) == self.messages_expected)

    def receive_message(self, message: Message):
        self.rec_messages.append(message)

    def messages_sent(self, message_count: int):
        self.messages_sent_count += message_count

    def __str__(self):
        return "Test Mock Event"


class TestKenningProtocol:
    PROTOCOL_EVENT_CLASS_PATH = (
        "kenning.protocols.kenning_protocol.ProtocolEvent"
    )

    @staticmethod
    def _assert_messages_equal(left: Message, right: Message):
        assert left.message_type == right.message_type
        assert left.payload == right.payload
        serializables_equal(left.flags, right.flags)
        assert left.flow_control_flags == right.flow_control_flags

    def _assert_message_lists_equal(
        self, left: List[Message], right: List[Message]
    ):
        assert len(left) == len(right)
        for i in range(len(left)):
            self._assert_messages_equal(left[i], right[i])

    EXAMPLE_MIXED_MESSAGE_STREAM = [
        Message(
            DEFAULT_MESSAGE_TYPE,
            b"\x3F\x4C",
            FlowControlFlags.TRANSMISSION,
            Flags(),
        ),
        Message(MessageType.MODEL, None),
        Message(
            DEFAULT_MESSAGE_TYPE,
            b"\5G\x12\x63",
            FlowControlFlags.REQUEST,
            Flags(
                {
                    FlagName.SPEC_FLAG_1: True,
                }
            ),
        ),
        Message(
            DEFAULT_MESSAGE_TYPE,
            b"\x3F\x4C",
            FlowControlFlags.ACKNOWLEDGE,
            Flags(
                {
                    FlagName.FIRST: True,
                }
            ),
        ),
    ]

    @pytest.mark.parametrize(
        "incoming_message_buffer, expected_dump_buffer",
        [
            (
                [EXAMPLE_MIXED_MESSAGE_STREAM[0]],
                [EXAMPLE_MIXED_MESSAGE_STREAM[0]],
            ),
            (
                EXAMPLE_MIXED_MESSAGE_STREAM[0:1],
                [EXAMPLE_MIXED_MESSAGE_STREAM[0]],
            ),
            (
                EXAMPLE_MIXED_MESSAGE_STREAM,
                [EXAMPLE_MIXED_MESSAGE_STREAM[0]]
                + EXAMPLE_MIXED_MESSAGE_STREAM[2:],
            ),
        ],
    )
    def test_receiver(
        self,
        protocol: KenningProtocol,
        incoming_message_buffer: List[Message],
        expected_dump_buffer: List[Message],
    ):
        (
            CONNECTION_PROTOCOL_SIDE,
            CONNECTION_OTHER_DEVICE_SIDE,
        ) = multiprocessing.Pipe(duplex=True)
        event_mock = MockEvent(DEFAULT_MESSAGE_TYPE, len(expected_dump_buffer))

        def kenning_protocol_receive_message_mock(timeout: float):
            if CONNECTION_PROTOCOL_SIDE.poll(timeout):
                return CONNECTION_PROTOCOL_SIDE.recv()
            else:
                return None

        protocol.receive_message = kenning_protocol_receive_message_mock

        protocol.start()

        protocol.current_protocol_events = {
            DEFAULT_MESSAGE_TYPE: event_mock,
        }

        for message in incoming_message_buffer:
            CONNECTION_OTHER_DEVICE_SIDE.send(message)

        while not event_mock.is_completed():
            pass

        protocol.stop()
        self._assert_message_lists_equal(
            expected_dump_buffer, event_mock.rec_messages
        )

    def test_send_messages(self, protocol: KenningProtocol):
        message_dump_buffer = []
        event_mock = MockEvent(MessageType.OUTPUT, 0)
        event_mock_2 = MockEvent(None, 0)
        event_mock_3 = MockEvent(MessageType.IO_SPEC, 0)
        event_mock_4 = MockEvent(MessageType.MODEL, 0)
        messages = [
            Message(
                MessageType.IO_SPEC,
                b"\x3F\x4C",
                FlowControlFlags.TRANSMISSION,
                Flags(),
            ),
            Message(MessageType.MODEL, None),
        ]

        def kenning_protocol_send_message_mock(message: Message):
            message_dump_buffer.append(message)

        protocol.send_message = kenning_protocol_send_message_mock

        protocol.start()

        # The 'send_messages' method should choose the proper event based
        # on matching the key with the message type passed as it's argument
        # (not message type from sent messages, nor the one stored in the
        # calling object)
        protocol.current_protocol_events = {
            None: event_mock,
            MessageType.OUTPUT: event_mock_2,
            MessageType.MODEL: event_mock_4,
            MessageType.IO_SPEC: event_mock_3,
        }
        protocol.send_messages(None, messages)
        while len(message_dump_buffer) != len(messages):
            pass
        protocol.stop()

        assert 0 == event_mock_2.messages_sent_count
        assert 0 == event_mock_3.messages_sent_count
        assert 0 == event_mock_4.messages_sent_count
        assert len(messages) == event_mock.messages_sent_count

        self._assert_message_lists_equal(messages, message_dump_buffer)

    @staticmethod
    def _assert_ougoing_transmission_success(
        message_type,
        message_number,
        payload,
        flags,
        queue,
        max_message_payload_size,
    ):
        assert message_number == queue.qsize()
        for i in range(message_number):
            message = queue.get()
            assert message_type == message.message_type
            if i == 0:
                assert message.flags.first
            if i == message_number - 1:
                assert message.flags.last
            for flag in FLAG_BINDINGS.keys():
                message_flag = getattr(message.flags, str(FLAG_BINDINGS[flag]))
                if flag in flags and flag.for_type(message_type):
                    assert message_flag
                else:
                    assert not message_flag
            assert message.flags.has_payload
            assert FlowControlFlags.TRANSMISSION == message.flow_control_flags
            assert (
                payload[
                    i * max_message_payload_size : (i + 1)
                    * max_message_payload_size
                ]
                == message.payload
            )

    @pytest.mark.parametrize(
        "message_type, payload_size, flags, max_message_size",
        [
            (
                MessageType.MODEL,
                1000,
                [TransmissionFlag.IS_HOST_MESSAGE, TransmissionFlag.SUCCESS],
                20,
            ),
            (
                MessageType.OUTPUT,
                1000,
                [
                    TransmissionFlag.IS_HOST_MESSAGE,
                    TransmissionFlag.SERIALIZED,
                ],
                1000,
            ),
            (
                MessageType.IO_SPEC,
                1000,
                [
                    TransmissionFlag.IS_HOST_MESSAGE,
                    TransmissionFlag.SERIALIZED,
                ],
                1300,
            ),
            (
                MessageType.MODEL,
                5968,
                [TransmissionFlag.IS_HOST_MESSAGE, TransmissionFlag.SUCCESS],
                18000,
            ),
            (
                MessageType.IO_SPEC,
                10000,
                [
                    TransmissionFlag.IS_HOST_MESSAGE,
                    TransmissionFlag.SERIALIZED,
                ],
                1241,
            ),
        ],
    )
    def test_transmit(
        self,
        protocol: KenningProtocol,
        message_type: MessageType,
        payload_size: int,
        flags: List[TransmissionFlag],
        max_message_size: int,
    ):
        protocol.max_message_size = max_message_size
        max_message_payload_size = max_message_size - HEADER_SIZE
        payload = bytes([i % 256 for i in range(payload_size)])
        message_number = ceil(len(payload) / max_message_payload_size)

        (
            CONNECTION_PROTOCOL_SIDE,
            CONNECTION_OTHER_DEVICE_SIDE,
        ) = multiprocessing.Pipe(duplex=True)

        SENT_MESSAGES = multiprocessing.Queue()

        def kenning_protocol_receive_message_mock(
            timeout: Optional[float] = None
        ):
            if CONNECTION_PROTOCOL_SIDE.poll(timeout):
                return CONNECTION_PROTOCOL_SIDE.recv()
            else:
                return None

        def kenning_protocol_send_message_mock(message: Message):
            CONNECTION_PROTOCOL_SIDE.send(message)

        def other_device_mock(connection, sent_messages):
            while True:
                message = connection.recv()
                sent_messages.put(message)
                if message.flags.last:
                    break

        protocol.send_message = kenning_protocol_send_message_mock
        protocol.receive_message = kenning_protocol_receive_message_mock
        other_device = multiprocessing.Process(
            target=other_device_mock,
            args=(CONNECTION_OTHER_DEVICE_SIDE, SENT_MESSAGES),
        )
        other_device.start()
        protocol.start()

        protocol.transmit_blocking(message_type, payload, flags)
        protocol.stop()
        other_device.terminate()
        self._assert_ougoing_transmission_success(
            message_type,
            message_number,
            payload,
            flags,
            SENT_MESSAGES,
            max_message_payload_size,
        )

        SENT_MESSAGES = multiprocessing.Queue()
        other_device = multiprocessing.Process(
            target=other_device_mock,
            args=(CONNECTION_OTHER_DEVICE_SIDE, SENT_MESSAGES),
        )
        other_device.start()
        protocol.start()

        protocol.transmit(message_type, payload, flags)
        while len(protocol.current_protocol_events) != 0:
            pass

        protocol.stop()
        other_device.terminate()

        self._assert_ougoing_transmission_success(
            message_type,
            message_number,
            payload,
            flags,
            SENT_MESSAGES,
            max_message_payload_size,
        )

    @pytest.mark.parametrize(
        "transmission, accepted, retry",
        [
            (MODEL_TRANSMISSION, False, 0),
            (IO_SPEC_TRANSMISSION_SINGLE_MESSAGE, False, 0),
            (IO_SPEC_TRANSMISSION_NO_PAYLOAD, False, 0),
            (MODEL_TRANSMISSION, True, 3),
            (IO_SPEC_TRANSMISSION_SINGLE_MESSAGE, True, 3435),
            (IO_SPEC_TRANSMISSION_NO_PAYLOAD, True, 1),
            (MODEL_TRANSMISSION, True, 0),
            (IO_SPEC_TRANSMISSION_SINGLE_MESSAGE, True, 15),
            (IO_SPEC_TRANSMISSION_NO_PAYLOAD, True, -4),
        ],
    )
    def test_request(
        self,
        protocol: KenningProtocol,
        transmission: Tuple[
            List[Message], MessageType, bytes, List[TransmissionFlag]
        ],
        accepted: bool,
        retry: int,
        random_byte_data: bytes,
    ):
        (
            CONNECTION_PROTOCOL_SIDE,
            CONNECTION_OTHER_DEVICE_SIDE,
        ) = multiprocessing.Pipe(duplex=True)
        messages, message_type, payload, flags = transmission

        REQUEST_MESSAGES = multiprocessing.Queue()

        def kenning_protocol_receive_message_mock(
            timeout: Optional[float] = None
        ):
            if CONNECTION_PROTOCOL_SIDE.poll(timeout):
                return CONNECTION_PROTOCOL_SIDE.recv()
            else:
                return None

        def kenning_protocol_send_message_mock(message: Message):
            CONNECTION_PROTOCOL_SIDE.send(message)

        protocol.send_message = kenning_protocol_send_message_mock
        protocol.receive_message = kenning_protocol_receive_message_mock
        success_callback_called = 0
        failure_callback_called = 0

        success_callback_message_type = None
        success_callback_payload = None
        success_callback_flags = None

        def success_callback(message_type, payload, flags):
            nonlocal success_callback_called
            nonlocal success_callback_message_type
            nonlocal success_callback_payload
            nonlocal success_callback_flags
            success_callback_called += 1
            success_callback_message_type = message_type
            success_callback_payload = payload
            success_callback_flags = flags

        deny_callback_message_type = None

        def deny_callback(event):
            nonlocal failure_callback_called
            nonlocal deny_callback_message_type
            failure_callback_called += 1
            deny_callback_message_type = event

        def other_device_mock(
            accepted,
            retry,
            message_type,
            messages,
            connection,
            request_messages,
        ):
            for _ in range(1 if accepted else retry + 1):
                while not connection.poll():
                    pass
                request_message = connection.recv()
                request_messages.put(request_message)
                if accepted:
                    for message in messages:
                        connection.send(message)
                else:
                    connection.send(
                        Message(
                            message_type,
                            None,
                            FlowControlFlags.ACKNOWLEDGE,
                            Flags(
                                {
                                    FlagName.FIRST: True,
                                    FlagName.LAST: True,
                                    FlagName.FAIL: True,
                                }
                            ),
                        )
                    )

        other_device = multiprocessing.Process(
            target=other_device_mock,
            args=(
                accepted,
                retry,
                message_type,
                messages,
                CONNECTION_OTHER_DEVICE_SIDE,
                REQUEST_MESSAGES,
            ),
        )
        other_device.start()
        protocol.start()
        response = protocol.request_blocking(
            message_type,
            success_callback,
            random_byte_data,
            [TransmissionFlag.IS_KENNING],
            1,
            retry,
            deny_callback,
        )
        protocol.stop()
        other_device.terminate()

        if accepted:
            assert 1 == success_callback_called
            assert not failure_callback_called
            assert message_type == success_callback_message_type
            assert payload == success_callback_payload
            assert flags == success_callback_flags
            assert payload == response[0]
            assert flags == response[1]
        else:
            assert 1 == failure_callback_called
            assert not success_callback_called
            assert message_type == deny_callback_message_type
            assert (None, None) == response

        failure_callback_called = 0
        success_callback_called = 0
        success_callback_message_type = None
        success_callback_payload = None
        success_callback_flags = None
        deny_callback_message_type = None
        other_device = multiprocessing.Process(
            target=other_device_mock,
            args=(
                accepted,
                retry,
                message_type,
                messages,
                CONNECTION_OTHER_DEVICE_SIDE,
                REQUEST_MESSAGES,
            ),
        )
        other_device.start()
        protocol.start()

        protocol.request(
            message_type,
            success_callback,
            random_byte_data,
            [TransmissionFlag.IS_KENNING],
            retry,
            deny_callback,
        )

        while len(protocol.current_protocol_events) != 0:
            pass

        protocol.stop()
        other_device.terminate()
        ProtocolEvent.callback_runner.close()
        ProtocolEvent.callback_runner.join()
        ProtocolEvent.callback_runner = ThreadPool(1)
        if accepted:
            assert 1 == success_callback_called
            assert not failure_callback_called
            assert message_type == success_callback_message_type
            assert payload == success_callback_payload
            assert flags == success_callback_flags
        else:
            assert 1 == failure_callback_called
            assert not success_callback_called
            assert message_type == deny_callback_message_type

        while not REQUEST_MESSAGES.empty():
            request_message = REQUEST_MESSAGES.get()
            assert request_message.flags.first
            assert request_message.flags.last
            assert request_message.flags.has_payload
            assert request_message.flags.is_kenning
            assert random_byte_data == request_message.payload
            assert (
                FlowControlFlags.REQUEST == request_message.flow_control_flags
            )
            assert message_type == request_message.message_type

    @pytest.mark.parametrize(
        "transmission, limit",
        [
            (IO_SPEC_TRANSMISSION_SINGLE_MESSAGE, 4),
            (MODEL_TRANSMISSION, 20),
            (IO_SPEC_TRANSMISSION_NO_PAYLOAD, 13),
        ],
    )
    def test_listen(
        self,
        protocol: KenningProtocol,
        transmission: Tuple[
            List[Message], MessageType, bytes, List[TransmissionFlag]
        ],
        limit: int,
        random_byte_data: bytes,
    ):
        (
            CONNECTION_PROTOCOL_SIDE,
            CONNECTION_OTHER_DEVICE_SIDE,
        ) = multiprocessing.Pipe(duplex=True)
        messages, message_type, payload, flags = transmission

        def kenning_protocol_receive_message_mock(
            timeout: Optional[float] = None
        ):
            if CONNECTION_PROTOCOL_SIDE.poll(timeout):
                return CONNECTION_PROTOCOL_SIDE.recv()
            else:
                return None

        def kenning_protocol_send_message_mock(message: Message):
            CONNECTION_PROTOCOL_SIDE.send(message)

        protocol.send_message = kenning_protocol_send_message_mock
        protocol.receive_message = kenning_protocol_receive_message_mock

        def other_device_wait():
            nonlocal protocol
            while message_type not in protocol.current_protocol_events.keys():
                pass

        def other_device_send_transmission(messages, connection):
            other_device_wait()
            for message in messages:
                connection.send(message)

        def other_device_send_request(messages, connection):
            other_device_wait()
            # To simulate a request we change flow control values in messages.
            request_messages = copy.deepcopy(messages)
            for message in request_messages:
                message.flow_control_flags = FlowControlFlags.REQUEST
                connection.send(message)

        transmission_count = 0
        request_count = 0

        def other_device_send_transmissions_and_requests(
            limit, messages, connection
        ):
            nonlocal transmission_count
            nonlocal request_count
            for i in range(limit):
                if i % 2 == 1:
                    other_device_send_transmission(messages, connection)
                    transmission_count += 1
                else:
                    other_device_send_request(messages, connection)
                    request_count += 1

        protocol.start()
        other_device = Thread(
            target=other_device_send_request,
            args=(messages, CONNECTION_OTHER_DEVICE_SIDE),
        )
        other_device.start()
        type, _message_type, _payload, _flags = protocol.listen_blocking(
            message_type, None, None, 1
        )
        other_device.join()
        assert IncomingEventType.REQUEST == type
        assert message_type == _message_type
        assert payload == _payload
        assert flags == _flags

        other_device = Thread(
            target=other_device_send_transmission,
            args=(messages, CONNECTION_OTHER_DEVICE_SIDE),
        )
        other_device.start()
        type, _message_type, _payload, _flags = protocol.listen_blocking(
            message_type, None, None, 1
        )
        other_device.join()
        assert payload == _payload
        assert flags == _flags
        assert message_type == _message_type
        assert IncomingEventType.TRANSMISSION == type

        protocol.stop()

        transmission_callback_called = 0
        request_callback_called = 0

        transmission_callback_message_type = []
        transmission_callback_payload = []
        transmission_callback_flags = []

        def transmission_callback(message_type, payload, flags):
            nonlocal transmission_callback_called
            nonlocal transmission_callback_message_type
            nonlocal transmission_callback_payload
            nonlocal transmission_callback_flags
            transmission_callback_called += 1
            transmission_callback_message_type.append(message_type)
            transmission_callback_payload.append(payload)
            transmission_callback_flags.append(flags)

        request_callback_message_type = []
        request_callback_payload = []
        request_callback_flags = []

        def request_callback(message_type, payload, flags):
            nonlocal request_callback_called
            nonlocal request_callback_message_type
            request_callback_called += 1
            request_callback_message_type.append(message_type)
            request_callback_payload.append(payload)
            request_callback_flags.append(flags)

        protocol.start()

        protocol.listen(
            message_type, transmission_callback, request_callback, limit
        )

        other_device = Thread(
            target=other_device_send_transmissions_and_requests,
            args=(limit, messages, CONNECTION_OTHER_DEVICE_SIDE),
        )
        other_device.start()

        while len(protocol.current_protocol_events) != 0:
            pass

        other_device.join()
        protocol.stop()
        ProtocolEvent.callback_runner.close()
        ProtocolEvent.callback_runner.join()
        ProtocolEvent.callback_runner = ThreadPool(1)

        assert limit == transmission_callback_called + request_callback_called
        assert transmission_count == transmission_callback_called
        assert request_count == request_callback_called

        for _message_type in transmission_callback_message_type:
            assert message_type == _message_type
        for _payload in transmission_callback_payload:
            assert payload == _payload
        for _flags in transmission_callback_flags:
            assert flags == _flags
        for _message_type in request_callback_message_type:
            assert message_type == _message_type
        for _flags in request_callback_flags:
            assert flags == _flags
        for _payload in request_callback_payload:
            assert payload == _payload
