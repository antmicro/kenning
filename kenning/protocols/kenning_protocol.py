# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of Kenning Protocol (a communication protocol for
exchanging inference data between devices).
"""

import time
from abc import ABC, abstractmethod
from enum import Enum
from math import ceil
from multiprocessing.pool import ThreadPool
from typing import (
    Callable,
    List,
    Tuple,
)

from kenning.protocols.bytes_based_protocol import BytesBasedProtocol
from kenning.protocols.message import (
    FlagName,
    Flags,
    FlowControlFlags,
    Message,
    MessageType,
)
from kenning.utils.event_with_args import EventWithArgs
from kenning.utils.logger import KLogger

MAX_MESSAGE_PAYLOAD_SIZE = 1024


class TransmissionFlag(Enum):
    """
    Flags sent in a transmission, that are available to the protocol user.

    Some flags are specific for 1 message type, 'message_type' property is
    storing this information ('None' means, that the flag is meant for any
    message type).


    * SUCCESS - Transmission is informing about a success.
    * FAIL - Transmission is informing about a failure. Payload is an error
      code sent by the other side.
    * IS_HOST_MESSAGE - Not set if the transmission was sent by the target
      device being evaluated. Set otherwise.
    * SERIALIZED - Flag available only for IO_SPEC transmissions. Denotes
      whether the model IO specifications are serialized.
    """

    SUCCESS = 1, None
    FAIL = 2, None
    IS_HOST_MESSAGE = 3, None
    SERIALIZED = 4, MessageType.IO_SPEC

    def __new__(cls, value, message_type):
        member = object.__new__(cls)
        member._value_ = value
        member.message_type = message_type
        return member

    def for_type(self, message_type: MessageType) -> bool:
        """
        Method checks, whether a given transmission flag is valid for a given
        message type (some flags are message type specific).

        Parameters
        ----------
        message_type: MessageType
            Message type to check.

        Returns
        -------
        bool
            True if the flag is valid for any message type or if the flag is
            valid for the message type passed. False if the flag is specific
            for a different message type.
        """
        return self.message_type is None or self.message_type == message_type


class ProtocolEvent(ABC):
    """
    Class representing a protocol flow control event (OutgoingRequest,
    Transmission etc.).

    Events operate as finite state machines - they are created by the main
    thread and are serviced and deleted by the receiver and transmitter
    threads. Transmitter is constantly checking, whether any event has
    messages to send. Receiver is waiting for messages and upon receiving
    a message, sends it to the proper event (mased on 'message_type').

    Every time an event receives or sends a message, it's state is changed
    until it reaches it's final state. Therefore every time after changing
    the Event's state, receiver/transmitter should check if the event is in
    it's final state (using the 'is_completed' method) and if so - remove it.
    """

    # Using 1 thread to prevent race conditions between callbacks.
    callback_runner = ThreadPool(1)

    class State(Enum):
        """
        Enum with states common for all types of Protocol Events.

        List of states and possible state transitions for each state:
        * NEW - Initial state - event created, but not active yet.
          Transitions:
            * <sub-class initial active state> - after 'start' or
              'start_blocking' call.
        * CONFIRMED - Event is completed successfully (final state).
        * DENIED - Event has failed (final state).
        """

        NEW = 0
        CONFIRMED = 1
        DENIED = 2

    # This has to be overridden by a subclass with a valid state
    INITIAL_ACTIVE_STATE = None

    def __init__(self, message_type: MessageType):
        self.state = self.State.NEW
        self.message_type = message_type
        self.run_callback_on_thread = None

    def _activate(self):
        """
        Sets the state to the initial active state (which means the event is
        not active and can receive/send messages).
        """
        if self.INITIAL_ACTIVE_STATE is None:
            raise NotImplementedError(
                f"Protocol Event {type(self)} has undefined initial active"
                " state and cannot be activated."
            )
        self.state = self.INITIAL_ACTIVE_STATE

    def start(
        self,
        success_callback: Callable["ProtocolEvent", None],
        deny_callback: Callable["ProtocolEvent", None],
    ):
        """
        Activates the ProtocolEvent in non-blocking mode, which means that if
        the event finishes successfully, a callback will be invoked on a
        whole new thread. Success callback will be invoked in protocol
        success (NOTE: even if, for example, the incoming transmission is a
        failure transmission (has the FAIL flag set) this is still protocol
        success, because the failure messages were successfully received).
        Deny callback will be called in case of protocol failure. To the
        callback a 'ProtocolEvent' child-class object will be passed (what
        class it will be - depends on the situation - look into docstrings
        of ProtocolEvents inheriting classes).

        Parameters
        ----------
        success_callback: Callable[ProtocolEvent, None]
            Callback to be invoked, if the event finishes successfully,
        deny_callback: Callable[ProtocolEvent, None]
            Callback to be invoked, if the event fails.
        """
        self.success_callback = success_callback
        self.deny_callback = deny_callback
        self.run_callback_on_thread = True
        self._activate()

    def start_blocking(self):
        """
        Activates the Protocol Event in blocking mode (which means, that the
        'wait' method may be then used to wait until the event is done).
        Alternatively, the function may be used to implement a busy wait
        mechanism (by polling the object with the 'is_completed' method).

        In blocking mode the object will not invoke callbacks. Instead values
        will be returned by the 'wait' method, this includes the object, that
        would normally be passed as callback argument.
        """
        self.completed_event = EventWithArgs()
        self.completed_event.clear()
        self.run_callback_on_thread = False
        self._activate()

    def wait(self) -> Tuple[bool, "ProtocolEvent"]:
        """
        Blocks the thread until the Protocol Event is completed. Can be used
        only after calling 'start_blocking'.

        Returns
        -------
        Tuple[bool, ProtocolEvent]
            Bool informing whether the protocol was successful, relevant
            ProtocolEvent object (look into docstrings of the relevant
            ProtocolEvent's inheriting class for details).

        Raises
        ------
        ValueError
            Protocol event started in non-blocking mode, or not started at all.
        """
        if self.run_callback_on_thread:
            raise ValueError(
                "Attempted to wait for a non-blocking protocol event:"
                f" {type(self)} of type {self.message_type}"
            )
        return self.completed_event.wait()

    def signal_callback(
        self,
        is_successful: bool,
        callback_argument: "ProtocolEvent",
    ):
        """
        Method, that is supposed to be called by the inheriting classes, to
        trigger the callback/wait mechanism.

        In non-blocking mode the function executes the proper callback (success
        callback or deny callback).

        In blocking mode it sends a signal to the waiting threads (ones that
        called the 'wait' method). With the signal is passed a boolean
        informing whether the protocol succeeded and the object, that would
        otherwise be passed as an argument to the callback (see 'wait' method
        for details).

        Parameters
        ----------
        is_successful: bool
            If true, success_callback will be called, if not.
        callback_argument: "ProtocolEvent"
            Argument to pass to the callback 'is_successful' is True (otherwise
            the message type will be passed)

        Raises
        ------
        ValueError
            Attempted to call the method before 'start' or 'start_blocking' was
            called on the object.
        """
        # None is the initial value of this field, methods starting the event
        # ('start' and 'start_blocking') set it to either true or false.
        # If the field is set to None, it means the event hasn't been
        # started yet.
        if self.run_callback_on_thread is None:
            raise ValueError(
                f"Attempted to trigger the callback/wait mechanism on a {self}"
                "  object, that wasn't yet started."
            )
        # To see whether we are in blocking or non-blocking mode, we check the
        # value that is set by 'start' and 'start_blocking' methods.
        if self.run_callback_on_thread:
            self.callback_runner.apply_async(
                self.success_callback if is_successful else self.deny_callback,
                (callback_argument,),
            )
        else:
            self.completed_event.set((is_successful, callback_argument))

    def has_succeeded(self) -> bool:
        """
        Checks whether the Event has completed successfully.

        Returns
        -------
        bool
            True if the event has completed successfully, False if the
            event has not yet completed, or failed.
        """
        return self.state == self.State.CONFIRMED

    def is_completed(self) -> bool:
        """
        Checks whether the Event has completed (successfully or not).

        Returns
        -------
        bool
            True if the event has completed, False if not.
        """
        return self.state == self.State.DENIED or self.has_succeeded()

    @abstractmethod
    def has_messages_to_send(self) -> bool:
        """
        Checks whether the Event has any pending messages to send.

        Returns
        -------
        bool
            True if there are pending messages to send, False otherwise.
        """
        ...

    @abstractmethod
    def accepts_messages(self) -> bool:
        """
        Checks whether the Event is waiting for any incoming messages.

        Returns
        -------
        bool
            True if the event is waiting for a message, False otherwise.
        """
        ...

    def get_next_message(self) -> Message:
        """
        Gets next message, that the Event wants to send. Base class
        method validates object state and should be called at the
        beginning of all overriding methods.

        Returns
        -------
        Message
            Message to send.

        Raises
        ------
        ValueError
            Event has no messages to send.
        """
        if not self.has_messages_to_send():
            raise ValueError(
                f"Attempted to send a message from a {self.message_type}"
                f"{self} with no messages to send"
            )

    def receive_message(self, message: Message):
        """
        Processes an incoming message. Base class method implementation
        validates the argument and should be called at the beginning
        of all overriding methods.

        Parameters
        ----------
        message: Message
            Incoming message.

        Raises
        ------
        ValueError
            Event is not waiting for a message or the message passed has
            an invalid message type.
        """
        if not self.accepts_messages():
            raise ValueError(
                f"Attempted to pass a message to a {self.message_type}"
                f"{self} with no messages to send"
            )
        if message.message_type != self.message_type:
            raise ValueError(
                f"Message of type {message.message_type} passed to a request"
                f" of type {self.message_type}."
            )

    def refresh(self):
        """
        Method called regularly on every protocol event by the transmitter
        thread, to trigger any potential event state changes without sending
        or receiving a message.
        """
        pass


class Transmission(ProtocolEvent, ABC):
    """
    Class representing a transmission.
    """

    # Used for translating internal message flags to transmission flags.
    FLAG_BINDINGS = {
        TransmissionFlag.SUCCESS: FlagName.SUCCESS,
        TransmissionFlag.FAIL: FlagName.FAIL,
        TransmissionFlag.IS_HOST_MESSAGE: FlagName.IS_HOST_MESSAGE,
        TransmissionFlag.SERIALIZED: FlagName.SPEC_FLAG_1,
    }


class OutgoingTransmission(Transmission):
    """
    Class representing a transmission, that is being sent.

    Objects passed to the callback/wait mechanism (see 'signal_callback' method
    in the base class for details):
    * OutgoingTransmission - Regardless of whether the event succeeded or not.
    """

    class State(Enum):
        """
        Enum with states specific to the outgoing transmission.

        List of states and possible state transitions for each state:
        * PENDING - Messages waiting to be sent.
          Transitions:
            * CONFIRMED - 'get_next_message' called N times (where N is the
              number of messages to send in the transmission). Callback/wait
              mechanism will be triggered (protocol success),
        """

        PENDING = 3

    State._member_map_.update(ProtocolEvent.State._member_map_)

    INITIAL_ACTIVE_STATE = State.PENDING

    def __init__(
        self,
        message_type: MessageType,
        payload: bytes,
        set_flags: List[TransmissionFlag],
    ):
        """
        Initializes the transmission, sets the initial state, divides the
        payload into messages.

        Parameters
        ----------
        message_type: MessageType
            Type of messages.
        payload: bytes
            Payload for the transmission.
        set_flags: List[TransmissionFlag]
            List of transmission flags, that will be passed to the other
            side (if flag is in the list it will be set to True, otherwise
            will be set to False).
        """
        super().__init__(message_type)
        self.messages_to_send = ceil(len(payload) / MAX_MESSAGE_PAYLOAD_SIZE)
        self.messages = [
            Message(
                self.message_type,
                payload[
                    MAX_MESSAGE_PAYLOAD_SIZE * i : MAX_MESSAGE_PAYLOAD_SIZE
                    * (i + 1)
                ],
                FlowControlFlags.TRANSMISSION,
                Flags(
                    dict(
                        {
                            str(FlagName.FIRST): i == 0,
                            str(FlagName.LAST): i
                            == (self.messages_to_send - 1),
                        },
                        # Attaching user-facing flags.
                        **{
                            str(self.FLAG_BINDINGS[flag]): flag.for_type(
                                message_type
                            )
                            for flag in set_flags
                        },
                    )
                ),
            )
            for i in range(self.messages_to_send)
        ]

    def get_next_message(self) -> Message:
        # Validating with the base class method.
        super().get_next_message()
        message = self.messages[len(self.messages) - self.messages_to_send]
        self.messages_to_send -= 1
        if self.messages_to_send == 0:
            self.state = self.State.CONFIRMED
            self.signal_callback(True, self)
        return message

    def has_messages_to_send(self) -> bool:
        return self.state == self.State.PENDING

    def receive_message(self, message: Message):
        super().receive_message(message)
        raise NotImplementedError

    def accepts_messages(self) -> bool:
        return False


class IncomingTransmission(Transmission):
    """
    Class representing a transmission, that is being received.

    Objects passed to the callback/wait mechanism (see 'signal_callback' method
    in the base class for details):
    * IncomingTransmission - Regardless of whether the protocol has succeeded
      or not (if it has, 'get_contents' method can be called on the object, to
      get received data).
    """

    class State(Enum):
        """
        Enum with all states specific to an incoming transmission.

        List of states and possible state transitions for each state:
        * RECEIVING - Receiving messages. First message should have the FIRST
          flag set. Payload from every message is added to the transmission
          payload. Flags from the first message are translated into user-facing
          transmission flags.
          Transitions:
            * CONFIRMED - Received a message with the LAST flag set to True.
              Callback/wait mechanism will be triggered (protocol success).
            * DENIED - Message was received, but next message didn't arrive
              within the given timeout (transition  is triggered by the
              'refresh' method call).  Callback/wait mechanism will be
              triggered (protocol  failure).
        """

        RECEIVING = 4

    State._member_map_.update(ProtocolEvent.State._member_map_)

    INITIAL_ACTIVE_STATE = State.RECEIVING

    def __init__(
        self,
        message_type: MessageType,
        timeout: float = -1.0,
    ):
        """
        Initializes the transmission, sets the initial state.

        Parameters
        ----------
        message_type: MessageType
            Type of messages.
        timeout: float
            Maximum time between messages in seconds. Negative number denotes
            infinite timeout.
        """
        super().__init__(message_type)
        self.timeout = timeout
        self.payload = b""
        self.last_message_timestamp = None

    def receive_message(self, message: Message):
        if self.message_type != message.message_type:
            raise ValueError(
                f"Message of type {message.message_type} passed to a"
                f" transmission of type {self.message_type}"
            )
        if self.state != self.State.RECEIVING:
            KLogger.error(
                "Other side started a new transmission before finishing"
                f" transmission of type: {self.message_type}."
            )
        # Saving the time when last message was received, so that we know when
        # timeout is violated (see the 'refresh' method).
        self.last_message_timestamp = time.perf_counter()
        # User facing transmission flags are taken from the first message, so
        # we are saving them for later to be translated by the 'get_contents'
        # method.
        if message.flags.first:
            self.flags = message.flags
        # We are appending message payload at the end of the transmission
        # payload (to combine payload for all messages).
        if message.flags.has_payload:
            self.payload += message.payload
        if message.flags.last:
            self.state = self.State.CONFIRMED
            self.signal_callback(True, self)

    def accepts_messages(self) -> bool:
        return self.state == self.State.RECEIVING

    def has_messages_to_send(self) -> bool:
        return False

    def get_contents(
        self
    ) -> Tuple[MessageType, bytes, List[TransmissionFlag]]:
        """
        Gets data received in the transmission (payload, flags and other data).

        Returns
        -------
        Tuple[MessageType, bytes, List[TransmissionFlag]]
            Information received in the transmission.

        Raises
        ------
        ValueError
            Method was called, before the transmission was fully received.
        """
        if self.state != self.State.CONFIRMED:
            raise ValueError(
                f"Attempted to unpack unfinished transmission of type:"
                f" {self.message_type}"
            )
        flags = []
        for field, _, _ in self.flags.serializable_fields:
            if (
                getattr(self.flags, str(field)) == 1
                and field in self.FLAG_BINDINGS.values()
            ):
                for key, value in self.FLAG_BINDINGS.items():
                    if value == field and key.for_type(self.message_type):
                        flags.append(key)
        return self.message_type, self.payload, flags

    def refresh(self):
        """
        Checks whether time since last message is larger than the timeout set.
        """
        if self.state == self.State.RECEIVING:
            if (
                self.timeout >= 0
                and (time.perf_counter() - self.last_message_timestamp)
                > self.timeout
            ):
                self.state = self.State.DENIED
                self.signal_callback(False, self)


class OutgoingRequest(ProtocolEvent):
    """
    Class representing a request being sent.

    Objects passed to the callback/wait mechanism (see 'signal_callback' method
    in the base class for details):
    * OutgoingRequest - In case of protocol failure.
    * IncomingTransmission - In case of protocol success. The 'get_contents'
      method can be called on the object, to get data received.
    """

    class State(Enum):
        """
        Enum with all states specific to an outgoing request.

        List of states and possible state transitions for each state:
        * PENDING - Request message is waiting to be sent.
          Transitions:
          * SENT - 'get_next_message' was called (and it returned the request
            message, which was presumably sent).
        * SENT - Request message has been sent.
          Transitions:
          * PENDING - either time since sending the request message has
            exceeded the set timeout (in which case the transition is
            triggered by a 'refresh' method call), or ACKNOWLEDGE message
            with the FAIL flag set was received. However the 'retry' value
            is still larger than 0 (in which case it will be decremented)
            or it is negative (which means infinite retries).
          * DENIED - either time since sending the request message has
            exceeded the set timeout (in which case the transition is
            triggered by a 'refresh' method call), or ACKNOWLEDGE message
            with the FAIL flag set was received. However the 'retry' value
            is set to 0. Callback/wait mechanism will be triggered (protocol
            failure).
          * ACCEPTED - A TRANSMISSION message was received (it's FIRST flag
            should be set to 1).
        * ACCEPTED - A Transmission message has been received, further
          communication managed by a 'IncomingTransmission' class object
          created inside this object.
          Transitions:
          * PENDING - After calling 'refresh' or passing a received message
            into the inner 'IncomingTransmission' object, the object changed
            state to DENIED. However the 'retry' value is still larger than
            0 (in which case it will be decremented or it is negative (which
            means infinite retries).
          * DENIED - After calling 'refresh' or passing a received message
            into the inner 'IncomingTransmission' object, the object changed
            state to DENIED. However the 'retry' value is set to 0.
            Callback/wait mechanism will be triggered (protocol failure).
          * CONFIRMED - After calling 'refresh' or passing a received message
            into the inner 'IncomingTransmission' object, the object changed
            state to CONFIRMED. Callback/wait mechanism will be triggered
            (protocol success).
        """

        PENDING = 4
        SENT = 5
        ACCEPTED = 6

    State._member_map_.update(ProtocolEvent.State._member_map_)

    INITIAL_ACTIVE_STATE = State.PENDING

    def __init__(
        self,
        message_type: MessageType,
        timeout: float = -1.0,
        retry: int = 0,
    ):
        """
        Initializes the request, sets the initial state, creates the request
        message.

        Parameters
        ----------
        message_type: MessageType
            Type of messages.
        timeout: float
            Timeout in seconds. Negative number denotes infinite timeout.
        retry: int
            How many times the request will be retried in case of failure. If
            a negative number is given, request will be retried indefinitely.
        """
        super().__init__(message_type)
        self.retry = retry
        self.timeout = timeout
        self.incoming_transmission = None
        self.request_sent_timestamp = None
        self.request_message = Message(
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

    def _deny(self):
        """
        Handle a request failure (either retry or not).
        """
        # Negative numbers denotes infinite retries.
        if self.retry < 0:
            self.state = self.State.PENDING
        elif self.retry > 0:
            self.state = self.State.PENDING
            self.retry -= 1
        else:
            self.state = self.State.DENIED
            # If not retrying, we signal the Callback/wait mechanism.
            self.signal_callback(False, self)

    def _poll_transmission(self):
        """
        Once the incoming transmission responding to a request starts, this
        class uses an IncomingTransmission class object to manage it. After
        any method call, that may update the state of that object, we need
        poll it, to check if it's in a final state.
        """
        if self.incoming_transmission.is_completed():
            if self.incoming_transmission.has_succeeded():
                self.state = self.State.CONFIRMED
                self.signal_callback(True, self.incoming_transmission)
            else:
                self._deny()

    def receive_message(self, message: Message):
        super().receive_message(message)
        if (
            message.flow_control_flags == FlowControlFlags.ACKNOWLEDGE
            and self.state == self.State.SENT
            and message.flags.fail
        ):
            # Negative ackowlegement is received, which means the other side
            # is denying our request.
            self._deny()
        elif message.flow_control_flags == FlowControlFlags.TRANSMISSION:
            if self.state == self.State.SENT:
                # Transmission was started, so the other side accepted our
                # request. We create an IncomingTransmission class object,
                # to handle the transmission.
                self.state = self.State.ACCEPTED
                self.incoming_transmission = IncomingTransmission(
                    self.message_type, self.timeout
                )
                # We start the incoming transmission in blocking mode, but not
                # calling wait, so the call is not actually blocking anything.
                # We do it because we don't want that object to do any
                # signalling or callbacks - we will poll it instead.
                self.incoming_transmission.start_blocking()
            # Passing the TRANSMISSION message to the internal
            # IncomingTransmission object.
            self.incoming_transmission.receive_message(message)
            # Receiving that message could change the state of the transmission
            # (for example if it's the last message) so we check and update our
            # state accordingly.
            self._poll_transmission()
        else:
            KLogger.error(
                f"Unexpected message: {message.message_type}"
                f" {message.flow_control_flags} (during a request)."
            )

    def accepts_messages(self) -> bool:
        return self.state in (self.State.SENT, self.State.ACCEPTED)

    def has_messages_to_send(self) -> bool:
        return self.state == self.State.PENDING or (
            self.state == self.State.ACCEPTED
            and self.incoming_transmission.has_messages_to_send()
        )

    def get_next_message(self) -> Message:
        super().get_next_message()
        if self.state == self.State.PENDING:
            # Request message is waiting to be sent.
            self.state = self.state.SENT
            self.request_sent_timestamp = time.perf_counter()
            return self.request_message
        if self.state == self.State.ACCEPTED:
            # The other side accepted our request, so communication
            # is managed by an internal IncomingTransmission class
            # object. If we have messages to send, then these messages
            # are coming from that object.
            return self.incoming_transmission.get_next_message()
            # This operation could change the state of the transmission
            # (for example if it's the last message) so we check and update our
            # state accordingly.
            self._poll_transmission()

    def refresh(self):
        """
        Sends the request message again if the timeout has been exceed. Passes
        the 'refresh' method call to the child object, if it exists.
        """
        if (
            self.state == self.State.SENT
            and self.timeout >= 0
            and self.timeout
            < (time.perf_counter() - self.request_sent_timestamp)
        ):
            self._deny()
        if self.state == self.State.ACCEPTED:
            self.incoming_transmission.refresh()
            # This operation could change the state of the transmission
            # (for example if it's the last message) so we check and update our
            # state accordingly.
            self._poll_transmission()


class IncomingRequest(ProtocolEvent):
    """
    Class representing an incoming request. It is only used by the Listen event
    to pass information about a received request to the callback.
    """

    INITIAL_ACTIVE_STATE = ProtocolEvent.State.CONFIRMED

    def __init__(self, message_type: MessageType):
        super().__init__(message_type)
        self.state = ProtocolEvent.State.CONFIRMED

    def receive_message(self, message: Message):
        raise NotImplementedError

    def accepts_messages(self) -> bool:
        return False

    def has_messages_to_send(self) -> bool:
        return False

    def get_next_message(self) -> Message:
        raise NotImplementedError


class Listen(ProtocolEvent):
    """
    Class representing the listening protocol event.

    Objects passed to the callback/wait mechanism (see 'signal_callback' method
    in the base class for details):
    * IncomingRequest - When a request message is received (protocol success).
    * IncomingTransmission - When a transmission is successfully received
      (protocol success), or a transmission starts, but cannot be successfully
      finished (protocol failure).

    Note: Object of this class can call 'success_callback' function multiple
    times during its life.
    """

    class State(Enum):
        """
        States specific to the Listen event.

        List of states and possible state transitions for each state:
        * LISTENING - The object is listening for messages of the given type
          (either a message starting a transmission or a request message).
          If a request message is received, callback/wait system will be
          triggered (the 'signal_callback' method) and the 'limit' value
          decremented, unless it is set to None.
          Transitions:
          * RECEIVING_TRANSMISSION - Message starting a transmission was
            received (this message should have the FIRST flag set to True).
            The transmission will be managed by a 'IncomingTransmission' class
            object created inside this object.
          * CONFIRMED - Request message was received and after decrementing
            the 'limit' value it is equal to 0.
        * RECEIVING_TRANSMISSION - Incoming transmission is ongoing. All
          messages received (and 'refresh' calls) are passed to the inner
          'IncomingTransmission' object ('self.incoming_transmission').
          Except request messages (those will be handled in the exact same
          way as in the LISTENING state).
          Transitions:
          * LISTENING - After passing a message (or a 'refresh') call to
            'self.incoming_transmission' it changed state to a final state
            (CONFIRMED or DENIED). Callback system will be triggered with
            that object. 'limit' value will be decremented, and as long as
            it is not 0 now (so greater than 0 or None), this 'Listen'
            object will return to LISTENING state.
          * CONFIRMED - The 'limit' value is equal to 1 and either a request
            message was received, or after passing a message (or a 'refresh')
            call to 'self.incoming_transmission' it changed state to a final
            state (CONFIRMED or DENIED). Callback/wait system will be triggered
            for either the IncomingRequest or the IncomingTransmission.
            NOTE: If the 'limit' is set to 1 and during an ongoing transmission
            a request is received, that transmission will be discarded.
        """

        LISTENING = 4
        RECEIVING_TRANSMISSION = 5

    State._member_map_.update(ProtocolEvent.State._member_map_)

    INITIAL_ACTIVE_STATE = State.LISTENING

    def __init__(
        self,
        message_type: MessageType,
        limit: int = -1,
        timeout: float = -1.0,
    ):
        """
        Initializes the request, sets the initial state, creates the request
        message.

        Parameters
        ----------
        message_type: MessageType
            Type of messages.
        limit: int
            How many transmissions or requests can the object receive before
            de-activating. Non-positive number denotes no limit.
        timeout: float
            Timeout in seconds. Negative number denotes infinite timeout.
        """
        super().__init__(message_type)
        self.limit = limit
        self.timeout = timeout

    def _decrement_limit(self):
        """
        If a receiving limit is set, it has to be decremented by calling this
        function, every time a transmission is finished or a request is
        received.
        """
        if self.limit > 0:
            self.limit -= 1
            if self.limit == 0:
                self.state = self.State.CONFIRMED

    def _poll_transmission(self):
        """
        When first message of the transmission is received, the object will
        create and IncomingTransmission class object to manage it. This
        function checks the state of that inner object, and updates the
        state of this (Listen) abject accordingly. It shall be called
        after any event, that could update the state in
        self.incoming_transmission.
        """
        if self.incoming_transmission.is_completed():
            self.signal_callback(
                self.incoming_transmission.has_succeeded(),
                self.incoming_transmission,
            )
            self.state = self.State.LISTENING
            self._decrement_limit()

    def start_blocking(self):
        """
        Listen class doesn't support limits other than 1 in blocking mode.
        This method calls the default start_blocking method from the base
        class (to start the object in blocing mode normally), but then
        checks if the limit is set to 1, and if not - updates it.
        """
        super().start_blocking()
        if self.limit != 1:
            KLogger.error(
                f"Started Protocol Listening in blocking mode with a limit: "
                f"{self.limit} (limits other than 1 are not supported in  "
                "blocking mode) adjusting the limit to 1..."
            )
            self.limit = 1

    def receive_message(self, message: Message):
        super().receive_message(message)
        if message.flow_control_flags == FlowControlFlags.REQUEST:
            # Regardless of the state, we always receive an incoming request
            # message and trigger caallback.
            self.signal_callback(True, IncomingRequest(self.message_type))
            self._decrement_limit()
        elif self.state == self.State.LISTENING:
            if message.flow_control_flags == FlowControlFlags.TRANSMISSION:
                self.state = self.State.RECEIVING_TRANSMISSION
                self.incoming_transmission = IncomingTransmission(
                    self.message_type, self.timeout
                )
                self.incoming_transmission.start_blocking()
                self.incoming_transmission.receive_message(message)
                self._poll_transmission()
            else:
                KLogger.warning(
                    f"Protocol listening for {self.message_type} has received"
                    f" an unexpected message: {message.FlowControlFlags}."
                    " Message has been discarded."
                )
        elif self.state == self.State.RECEIVING_TRANSMISSION:
            self.incoming_transmission.receive_message(message)
            self._poll_transmission()

    def accepts_messages(self):
        return (
            self.state == self.State.RECEIVING_TRANSMISSION
            and self.incoming_transmission.accepts_messages()
        ) or self.state == self.State.LISTENING

    def has_messages_to_send(self):
        return (
            self.state == self.State.RECEIVING_TRANSMISSION
            and self.incoming_transmission.has_messages_to_send()
        )

    def get_next_message(self):
        super().get_next_message()
        return self.incoming_transmission.get_next_message()
        self._poll_transmission()

    def refresh(self):
        """
        Cancels the incoming transmissing if it's timeout has been exceeded.
        """
        if self.state == self.State.RECEIVING_TRANSMISSION:
            self.incoming_transmission.refresh()
            self._poll_transmission()


class KenningProtocol(BytesBasedProtocol, ABC):
    """
    Class for managing the flow of Kenning Protocol (unimplemented yet).
    """

    ...
