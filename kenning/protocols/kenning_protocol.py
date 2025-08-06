# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of Kenning Protocol (a communication protocol for
exchanging inference data between devices).
"""

import selectors
import time
from abc import ABC, abstractmethod
from enum import Enum
from math import ceil
from multiprocessing.pool import ThreadPool
from threading import Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple

from kenning.protocols.bytes_based_protocol import (
    BytesBasedProtocol,
    IncomingEventType,
    ProtocolFailureCallback,
    ProtocolSuccessCallback,
    TransmissionFlag,
)
from kenning.protocols.message import (
    FlagName,
    Flags,
    FlowControlFlags,
    Message,
    MessageType,
)
from kenning.utils.event_with_args import EventWithArgs
from kenning.utils.logger import KLogger

MAX_MESSAGE_PAYLOAD_SIZE = 1024 * 1024

# Used for translating internal message flags to transmission flags.
FLAG_BINDINGS = {
    TransmissionFlag.SUCCESS: FlagName.SUCCESS,
    TransmissionFlag.FAIL: FlagName.FAIL,
    TransmissionFlag.IS_HOST_MESSAGE: FlagName.IS_HOST_MESSAGE,
    TransmissionFlag.IS_ZEPHYR: FlagName.IS_ZEPHYR,
    TransmissionFlag.IS_KENNING: FlagName.IS_KENNING,
    TransmissionFlag.SERIALIZED: FlagName.SPEC_FLAG_1,
}


class ProtocolNotStartedError(Exception):
    """
    Exception raised by the protocol, when attempting to use a protocol
    object, that is not initialized/started.
    """


class ProtocolEvent(ABC):
    """
    Class representing a protocol flow control event (OutgoingRequest,
    Transmission etc.).

    Events operate as finite state machines - they are stored in a Dict
    in the KenningProtocol class.

    Each event has to be started ('start_blocking' or 'start' method).

    Receiver thread is waiting for messages and upon receiving a message,
    sends it to the proper event (based on 'message_type'), it is done
    by calling the 'receive_message' method, which then changes the state
    of the event, based on the current state and the message received.

    When an event wants to send messages, it calls the 'send_messages'
    method in KenningProtocol. After the messages are sent, 'messages_sent'
    method is called on the appropriate event (based on 'message_type').
    This is done on a separate worker thread (transmitter).

    Every time an event receives or sends a message, its state is changed
    until it reaches it's final state, when it should trigger the callback
    system, by calling 'signal_callback'.

    In some cases events may contain another events (child events). In such
    a case, methods like 'receive_message' and 'messages_sent' are called on
    the main event, which then should call it on its child events.
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

    def __init__(
        self, message_type: Optional[MessageType], protocol: "KenningProtocol"
    ):
        self.state = self.State.NEW
        self.message_type = message_type
        self.run_callback_on_thread = None
        self.lock = Lock()
        self.protocol = protocol

    def _activate(self):
        """
        Sets the state to the initial active state (which means the event is
        now active and can receive/send messages).
        """
        if self.INITIAL_ACTIVE_STATE is None:
            raise NotImplementedError(
                f"Protocol Event {self} has undefined initial active"
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
        # Setting arguments, that will be returned by the wait method of the
        # event, in case timeout is reached.
        self.completed_event = EventWithArgs((False, self))
        self.completed_event.clear()
        self.run_callback_on_thread = False
        self._activate()

    def wait(self, timeout: Optional[float]) -> Tuple[bool, "ProtocolEvent"]:
        """
        Blocks the thread until the Protocol Event is completed. Can be used
        only after calling 'start_blocking'.

        Parameters
        ----------
        timeout: Optional[float]
            Waiting timeout in seconds or None, which denotes infinite timeout.

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
                f"Attempted to wait for a non-blocking protocol event: {self}"
            )
        return self.completed_event.wait(timeout)

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
            Argument to pass to the callback.

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
                "Attempted to trigger the callback/wait mechanism on a"
                f" {self} object, that wasn't yet started."
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
    def accepts_messages(self) -> bool:
        """
        Checks whether the Event is waiting for any incoming messages.

        Returns
        -------
        bool
            True if the event is waiting for a message, False otherwise.
        """
        ...

    @abstractmethod
    def messages_sent(self, message_count: int):
        """
        Informs the object, that its messages has been sent (after
        it called 'send_messages' method). Updates state of the
        object accordingly.

        Parameters
        ----------
        message_count: int
            Number of messages sent from this event.
        """
        ...

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
                f"Attempted to pass a message to a {self}, that is not"
                " waiting for messages."
            )
        if message.message_type != self.message_type:
            raise ValueError(
                f"Message type {message.message_type} passed to a {self}"
            )

    def __str__(self):
        return f"{self.__class__.__name__}({self.message_type})"


class IncomingEvent(ProtocolEvent, ABC):
    """
    Abstract class for all Protocol Events, that may return payload and
    user-facing flags.
    """

    def __init__(
        self,
        message_type: MessageType,
        protocol: "KenningProtocol",
    ):
        super().__init__(message_type, protocol)
        self.payload = b""
        self.flags = Flags()

    def get_contents(
        self
    ) -> Tuple[MessageType, bytes, List[TransmissionFlag]]:
        """
        Gets data received in the event (payload, flags and other data).

        Returns
        -------
        Tuple[MessageType, bytes, List[TransmissionFlag]]
            Information received in the event.

        Raises
        ------
        ValueError
            Method was called, before the event was fully completed.
        """
        if self.state != self.State.CONFIRMED:
            raise ValueError(f"Attempted to unpack unfinished {self}.")
        flags = []
        for field, _, _ in self.flags.serializable_fields:
            if (
                getattr(self.flags, str(field)) == 1
                and field in FLAG_BINDINGS.values()
            ):
                for key, value in FLAG_BINDINGS.items():
                    if value == field and key.for_type(self.message_type):
                        flags.append(key)
        return self.message_type, self.payload, flags


class OutgoingEvent(ProtocolEvent, ABC):
    """
    Abstract class for all Protocol Events, that involve sending messages
    with payload and flags.
    """

    @staticmethod
    def translate_flags(
        message_type: MessageType, flags: List[TransmissionFlag]
    ) -> Dict[FlagName, bool]:
        """
        Converts a list of user-facing TransmissionFlags to a Dict of
        message flags (FlagName enum).

        Parameters
        ----------
        message_type: MessageType
            Message type for the transmission flags (this is relevant,
            because some flags are only for a specific message type).
        flags: List[TransmissionFlag]
            Transmission flags, that are supposed to be set.

        Returns
        -------
        Dict[FlagName, bool]
            A dict of all message flags, that were in the list.
        """
        return {
            str(FLAG_BINDINGS[flag]): True
            for flag in flags
            if flag.for_type(message_type)
        }


class OutgoingTransmission(OutgoingEvent):
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
        * PENDING - Called 'send_messages' on KenningProtocol.
          Transitions:
            * CONFIRMED - 'messages_sent' called with the N value (where N is
              the number of messages to send in the transmission).
              Callback/wait mechanism will be triggered (protocol success),
        """

        PENDING = 3

    State._member_map_.update(ProtocolEvent.State._member_map_)

    INITIAL_ACTIVE_STATE = State.PENDING

    def __init__(
        self,
        message_type: MessageType,
        protocol: "KenningProtocol",
        payload: Optional[bytes],
        set_flags: List[TransmissionFlag],
    ):
        """
        Initializes the transmission, sets the initial state, divides the
        payload into messages.

        Parameters
        ----------
        message_type: MessageType
            Type of messages.
        protocol: KenningProtocol
            A KenningProtocol instance, that this object was spawned by.
        payload: Optional[bytes]
            Payload for the transmission or None, if there is no payload.
        set_flags: List[TransmissionFlag]
            List of transmission flags, that will be passed to the other
            side (if flag is in the list it will be set to True, otherwise
            will be set to False).
        """
        super().__init__(message_type, protocol)
        if payload is None:
            payload = b""
        self.messages_to_send = max(
            1, ceil(len(payload) / MAX_MESSAGE_PAYLOAD_SIZE)
        )
        user_facing_flags = self.translate_flags(message_type, set_flags)
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
                        **user_facing_flags,
                    )
                ),
            )
            for i in range(self.messages_to_send)
        ]

    def _activate(self):
        """
        Sets the state to the initial active state (which means the event is
        not active and can receive/send messages). Gives KenningProtocol
        messages to send.
        """
        super()._activate()
        self.protocol.send_messages(self.message_type, self.messages)

    def messages_sent(self, message_count: int):
        if self.state == self.State.PENDING:
            self.messages_to_send -= message_count
            if self.messages_to_send == 0:
                self.state = self.State.CONFIRMED
                self.signal_callback(True, self)
            elif self.messages_to_send < 0:
                raise ValueError(
                    f"Invalid number of messages sent: {message_count}"
                    f" (should be no more than {self.messages_to_send})"
                )

    def receive_message(self, message: Message):
        raise NotImplementedError

    def accepts_messages(self) -> bool:
        return False


class IncomingTransmission(IncomingEvent):
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
        """

        RECEIVING = 4

    State._member_map_.update(ProtocolEvent.State._member_map_)

    INITIAL_ACTIVE_STATE = State.RECEIVING

    def __init__(
        self,
        message_type: MessageType,
        protocol: "KenningProtocol",
    ):
        """
        Initializes the transmission, sets the initial state.

        Parameters
        ----------
        message_type: MessageType
            Type of messages.
        protocol: KenningProtocol
            A KenningProtocol instance, that this object was spawned by.
        """
        super().__init__(message_type, protocol)

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

    def messages_sent(self, message_count: int):
        raise NotImplementedError


class OutgoingRequest(OutgoingEvent):
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
        * PENDING - Called 'messages_sent' on KenningProtocol with the
          request message, waiting for sending to complete.
          Transitions:
          * SENT - 'messages_sent' was called with 1 as argument (informing
            that the request message has been sent).
        * SENT - Request message has been sent.
          Transitions:
          * PENDING - ACKNOWLEDGE message with the FAIL flag set was
            received. However the 'retry' value is still larger than 0
            (in which case it will be decremented) or it is negative
            (which means infinite retries).
          * DENIED - ACKNOWLEDGE message with the FAIL flag set was received.
            However the 'retry' value is set to 0. Callback/wait mechanism
            will be triggered (protocol failure).
          * ACCEPTED - A TRANSMISSION message was received (it's FIRST flag
            should be set to 1).
        * ACCEPTED - A Transmission message has been received, further
          communication managed by a 'IncomingTransmission' class object
          created inside this object.
          Transitions:
          * PENDING - After passing a received message into the inner
            'IncomingTransmission' object, that object changed state to DENIED.
            However the 'retry' value is still larger than 0 (in which case
            it will be decremented or it is negative (which means infinite
            retries).
          * DENIED - After passing a received message into the inner
            'IncomingTransmission' object, that object changed state to DENIED.
            However the 'retry' value is set to 0.
            Callback/wait mechanism will be triggered (protocol failure).
          * CONFIRMED - After passing a received message into the inner
            'IncomingTransmission' object, it changed state to CONFIRMED.
            Callback/wait mechanism will be triggered (protocol success).
        """

        PENDING = 4
        SENT = 5
        ACCEPTED = 6

    State._member_map_.update(ProtocolEvent.State._member_map_)

    INITIAL_ACTIVE_STATE = State.PENDING

    def __init__(
        self,
        message_type: MessageType,
        protocol: "KenningProtocol",
        retry: int = 0,
        payload: Optional[bytes] = None,
        set_flags: List[TransmissionFlag] = [],
    ):
        """
        Initializes the request, sets the initial state, creates the request
        message.

        Parameters
        ----------
        message_type: MessageType
            Type of messages.
        protocol: KenningProtocol
            A KenningProtocol instance, that this object was spawned by.
        retry: int
            How many times the request will be retried in case of failure. If
            a negative number is given, request will be retried indefinitely.
        payload: Optional[bytes]
            Payload to send with the request message, or None if there is to
            be no payload.
        set_flags: List[TransmissionFlag]
            User-facing flags, that are to be sent with the request message.
        """
        super().__init__(message_type, protocol)
        self.retry = retry
        self.incoming_transmission = None
        user_facing_flags = self.translate_flags(message_type, set_flags)
        self.request_message = Message(
            message_type,
            payload,
            FlowControlFlags.REQUEST,
            Flags(
                dict(
                    {
                        FlagName.FIRST: True,
                        FlagName.LAST: True,
                    },
                    **user_facing_flags,
                )
            ),
        )

    def _send_request(self):
        self.state = self.State.PENDING
        self.protocol.send_messages(self.message_type, [self.request_message])

    def _activate(self):
        super()._activate()
        self._send_request()

    def _deny(self):
        """
        Handle a request failure (either retry or not).
        """
        # Negative numbers denotes infinite retries.
        if self.retry < 0:
            self._send_request()
        elif self.retry > 0:
            self.retry -= 1
            self._send_request()
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
                    self.message_type, self.protocol
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
                f"Unexpected message: {message.message_type},"
                f" {message.flow_control_flags} (during a request)."
            )

    def accepts_messages(self) -> bool:
        return self.state in (self.State.SENT, self.State.ACCEPTED)

    def messages_sent(self, message_count: int):
        if self.state == self.State.PENDING:
            if message_count != 1:
                raise ValueError(
                    f"Invalid message count: {message_count}"
                    " (only 1 request message at a time should be sent)."
                )
            self.state = self.state.SENT
        if self.state == self.State.ACCEPTED:
            # The other side accepted our request, so communication
            # is managed by an internal IncomingTransmission class
            # object.
            self.incoming_transmission.messages_sent(message_count)
            self._poll_transmission()


class IncomingRequest(IncomingEvent):
    """
    Class representing an incoming request.

    Objects passed to the callback/wait mechanism (see 'signal_callback' method
    in the base class for details):
    * IncomingRequest - When a request message is received (protocol success).
    """

    class State(Enum):
        """
        States specific to the Listen event.

        List of states and possible state transitions for each state:
        * RECEIVING - Waiting for the request message.
          Transitions:
          * CONFIRMED - 'receive_message' was called with a valid request
            message.
        """

        RECEIVING = 4

    State._member_map_.update(ProtocolEvent.State._member_map_)

    INITIAL_ACTIVE_STATE = State.RECEIVING

    def __init__(self, message_type: MessageType, protocol: "KenningProtocol"):
        super().__init__(message_type, protocol)

    def receive_message(self, message: Message):
        super().receive_message(message)
        if message.flow_control_flags == FlowControlFlags.REQUEST:
            self.payload = message.payload
            self.flags = message.flags
            self.state = self.State.CONFIRMED
            self.signal_callback(True, self)
        else:
            KLogger.error("Non-request message passed to an Incoming Request")

    def accepts_messages(self) -> bool:
        return self.state == self.State.RECEIVING

    def messages_sent(self, message_count: int):
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
          messages received (and 'messages_sent' calls) are passed to the
          inner 'IncomingTransmission' object ('self.incoming_transmission').
          Except request messages (those will be handled in the exact same
          way as in the LISTENING state).
          Transitions:
          * LISTENING - After passing a message to 'self.incoming_transmission'
            or calling 'messages_sent', it changed state to a final state
            (CONFIRMED or DENIED). Callback system will be triggered with
            that object. 'limit' value will be decremented, and as long as
            it is not 0 now (so greater than 0 or None), this 'Listen'
            object will return to LISTENING state.
          * CONFIRMED - The 'limit' value is equal to 1 and either a request
            message was received, or after passing a message to
            'self.incoming_transmission' it changed state to a final state
            (CONFIRMED or DENIED). Callback/wait system will be triggered
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
        message_type: Optional[MessageType],
        protocol: "KenningProtocol",
        limit: int = -1,
    ):
        """
        Initializes the request, sets the initial state, creates the request
        message.

        Parameters
        ----------
        message_type: Optional[MessageType]
            Type of messages to listen for. If set to None, it will listen
            for any message type (and infer type from the first incoming
            message)
        protocol: KenningProtocol
            A KenningProtocol instance, that this object was spawned by.
        limit: int
            How many transmissions or requests can the object receive before
            de-activating. Non-positive number denotes no limit.
        """
        super().__init__(message_type, protocol)
        self.limit = limit

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
        # If no type was specified, we infer type from the first arriving
        # message.
        if message.flow_control_flags == FlowControlFlags.REQUEST:
            # Regardless of the state, we always receive an incoming request
            # message and trigger caallback.
            incoming_request = IncomingRequest(
                message.message_type, self.protocol
            )
            incoming_request.start_blocking()
            incoming_request.receive_message(message)
            self.signal_callback(True, incoming_request)
            self._decrement_limit()
        elif self.state == self.State.LISTENING:
            if message.flow_control_flags == FlowControlFlags.TRANSMISSION:
                self.state = self.State.RECEIVING_TRANSMISSION
                self.incoming_transmission = IncomingTransmission(
                    message.message_type, self.protocol
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

    def messages_sent(self, message_count: int):
        if self.state == self.State.RECEIVING_TRANSMISSION:
            self.incoming_transmission.messages_sent(message_count)
        else:
            raise ValueError("Unexpected messages sent from this object.")


class KenningProtocol(BytesBasedProtocol, ABC):
    """
    Class for managing the flow of Kenning Protocol.
    """

    def __init__(
        self,
        timeout: int = -1,
    ):
        self.selector = selectors.DefaultSelector()
        self.input_buffer = b""
        self.current_protocol_events = {}
        self.event_lock = Lock()
        self.receiver_thread = None
        self.transmitter = None
        self.protocol_running = False
        super().__init__(timeout)

    def start(self):
        """
        Starts the protocol, creates and runs threads for receiving and
        transmitting messages.
        """
        if self.receiver_thread is None:
            self.input_buffer = b""
            self.current_protocol_events = {}
            self.protocol_running = True
            self.receiver_thread = Thread(target=self.receiver, args=())
            self.receiver_thread.start()
            self.transmitter = ThreadPool(1)
        else:
            ValueError("Protocol already started.")

    def stop(self):
        """
        Stops the protocol and joins all threads.
        """
        self.protocol_running = False
        if (
            self.receiver_thread is not None
            and self.receiver_thread.is_alive()
        ):
            self.receiver_thread.join()
        self.receiver_thread = None
        if self.transmitter is not None:
            self.transmitter.close()
            self.transmitter.terminate()
            self.transmitter = None

    def __del__(self):
        self.stop()

    def send_message(self, message: Message) -> bool:
        """
        Sends message to the target device.

        Parameters
        ----------
        message : Message
            Message to be sent.

        Returns
        -------
        bool
            True if succeeded.
        """
        KLogger.debug(f"Sending message {message}")
        ret = self.send_data(message.to_bytes())
        if not ret:
            KLogger.error(f"Error sending message {message}")
        return ret

    def receive_message(
        self, timeout: Optional[float] = None
    ) -> Optional[Message]:
        """
        Waits for incoming data from the other side of connection.

        This method should wait for the input data to arrive and return the
        appropriate status code along with received data.

        Parameters
        ----------
        timeout : Optional[float]
            Receive timeout in seconds. If timeout > 0, this specifies the
            maximum wait time, in seconds. If timeout <= 0, the call won't
            block, and will report the currently ready file objects. If timeout
            is None, the call will block until a monitored file object becomes
            ready.

        Returns
        -------
        Optional[Message]
            Received message, or None if message was not received.
        """
        while True:
            message, data_parsed, checksum_valid = Message.from_bytes(
                self.input_buffer
            )
            if message is not None:
                self.input_buffer = self.input_buffer[data_parsed:]
                KLogger.debug(f"Received message {message}")
                return message
            if timeout <= 0:
                return None
            if timeout is None or timeout > 0:
                data = self.gather_data(timeout)
                if data is not None:
                    self.input_buffer += data
                if data is None and timeout is not None:
                    return None

    @abstractmethod
    def send_data(self, data: Any) -> bool:
        """
        Sends data to the target device.

        Data can be model to use, input to process, additional configuration.

        Parameters
        ----------
        data : Any
            Data to send.

        Returns
        -------
        bool
            True if successful.
        """
        ...

    @abstractmethod
    def receive_data(self, connection: Any, mask: int) -> Optional[Any]:
        """
        Receives data from the target device.

        Parameters
        ----------
        connection : Any
            Connection used to read data.
        mask : int
            Selector mask from the event.

        Returns
        -------
        Optional[Any]
            Status of receive and optionally data that was received.
        """
        ...

    def gather_data(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """
        Gathers data from the client.

        This method should be called by receive_message in order to get data
        from the client.

        Parameters
        ----------
        timeout : Optional[float]
            Receive timeout in seconds. If timeout > 0, this specifies the
            maximum wait time, in seconds. If timeout <= 0, the call won't
            block, and will report the currently ready file objects. If timeout
            is None, the call will block until a monitored file object becomes
            ready.

        Returns
        -------
        Optional[bytes]
            Received data.
        """
        start_time = time.perf_counter()
        while True:
            events = self.selector.select(timeout=timeout)
            results = b""
            for key, mask in events:
                if mask & selectors.EVENT_READ:
                    callback = key.data
                    data = callback(key.fileobj, mask)
                    if data is not None:
                        results += data
            if results:
                return results
            elif not timeout or (time.perf_counter() - start_time > timeout):
                return None

    def receiver(self):
        """
        Method sitting in a loop, receiving messages and passing them to the
        relevant ProtocolEvent class object in the 'current_protocol_events'
        dict, based on message type.

        This method is meant to be running constantly on a separate thread, if
        the protocol is active.
        """
        while self.protocol_running:
            message = self.receive_message(1)
            if message is not None:
                relevant_event = None
                if message.message_type in self.current_protocol_events.keys():
                    relevant_event = self.current_protocol_events[
                        message.message_type
                    ]
                # If the message type doesn't directly match any event, we
                # check if a None type event (so an event acceting any message
                # type) is in the dict.
                elif None in self.current_protocol_events:
                    relevant_event = self.current_protocol_events[None]
                if relevant_event is not None:
                    with relevant_event.lock:
                        if relevant_event.accepts_messages():
                            relevant_event.receive_message(message)

    def send_messages(
        self, message_type: MessageType, messages: List[Message]
    ):
        """
        Gives a job to the 'transmitter' thread pool, that will send messages
        and call 'messages_sent' method on the appropriate ProtocolEvent
        object in the 'self.current_protocol_events' dict (based on message
        type).

        This is meant be be called by the ProtocolEvent objects to send
        messages and update their state.

        Parameters
        ----------
        message_type: MessageType
            Message type of the ProtocolEvent, that sends the messages. This
            is needed, so that we can later find that object in the
            'self.current_protocol_events' dict and call 'messages_sent'.
            We could not use a reference here, because for example if
            a Listen object in the dict has an IncomingTransmission object
            inside, and that inner object calls this method, we need to call
            'message_sent' on the whole Listen object, not just the inner one.
        messages: List[Message]
            List of messages to send.
        """

        def sender():
            relevant_event = None
            if message_type in self.current_protocol_events.keys():
                relevant_event = self.current_protocol_events[message_type]
            elif None in self.current_protocol_events:
                relevant_event = self.current_protocol_events[None]
            if relevant_event is not None:
                with relevant_event.lock:
                    for message in messages:
                        self.send_message(message)
                    relevant_event.messages_sent(len(messages))
            else:
                KLogger.warning(
                    "Event that attempted to send messages no longer exists."
                )

        self.transmitter.apply_async(sender)

    def run_event_blocking(
        self,
        event: ProtocolEvent,
        timeout: Optional[float],
    ) -> Tuple[bool, ProtocolEvent]:
        """
        Adds a protocol event to the 'current_protocol_events' dict, so that
        it will be serviced by the receiver thread, starts it in blocking
        mode and blocks the current thread until it completes.

        Parameters
        ----------
        event: ProtocolEvent
            Event to start.
        timeout: Optional[float]
            Maximum blocking time in seconds (or None for infinite timeout).

        Returns
        -------
        Tuple[bool, ProtocolEvent]
            True if event succeeded, False if not, ProtocolEvent object
            returned by the event (see the dostring of the relevant
            ProtocolEvent to see what object will be passed here).

        Raises
        ------
        ValueError
            Attempted to start the event, while another event of the same
            type was already in progress.
        ProtocolNotStartedError
            Protocol is not active, call 'start' first.
        """
        if not self.protocol_running:
            raise ProtocolNotStartedError("Protocol not started.")
        event_started = False
        with self.event_lock:
            if event.message_type not in self.current_protocol_events.keys():
                self.current_protocol_events[event.message_type] = event
                event.start_blocking()
                KLogger.debug(f"{event} has been started in blocking mode.")
                event_started = True
        if event_started:
            return event.wait(timeout)
        else:
            raise ValueError(
                f"{event} attempted to start while another event of the same"
                " type was in progress."
            )

    def run_event(
        self,
        event: ProtocolEvent,
        success_callback: Callable[ProtocolEvent, None],
        deny_callback: Callable[ProtocolEvent, None],
    ):
        """
        Adds a protocol event to the 'current_protocol_events' dict, so that
        it will be serviced by the receiver thread, starts it in non-blocking
        mode and passes callbacks.

        Parameters
        ----------
        event: ProtocolEvent
            Event to start.
        success_callback: Callable[ProtocolEvent, None]
            Function, that will be called if the event succeeds.
        deny_callback: Callable[ProtocolEvent, None]
            Function, that will be called if the event fails.

        Raises
        ------
        ValueError
            Attempted to start the event, while another event of the same
            type was already in progress.
        ProtocolNotStartedError
            Protocol is not active, call 'start' first.
        """
        if not self.protocol_running:
            raise ProtocolNotStartedError("Protocol not started.")
        with self.event_lock:
            if event.message_type not in self.current_protocol_events.keys():
                self.current_protocol_events[event.message_type] = event
                event.start(success_callback, deny_callback)
                KLogger.debug(
                    f"{event} has been started in non-blocking mode."
                )
            else:
                raise ValueError(
                    f"{event} attempted to start while another event of the"
                    " same type was in progress."
                )

    def finish_event(self, event: ProtocolEvent):
        """
        Removes an event from the 'current_protocol_events' dict and
        logs it's success or failure.

        Parameters
        ----------
        event: ProtocolEvent
            Event to finish
        """
        if event.has_succeeded():
            KLogger.debug(f"{event} has succeeded.")
        else:
            KLogger.error(f"{event} has failed.")
        with self.event_lock:
            if event.message_type in self.current_protocol_events.keys():
                self.current_protocol_events.pop(event.message_type)

    def transmit(
        self,
        message_type: MessageType,
        payload: Optional[bytes] = None,
        flags: List[TransmissionFlag] = [],
        failure_callback: Optional[ProtocolFailureCallback] = None,
    ):
        def handle_success(event: ProtocolEvent):
            self.finish_event(event)

        def handle_failure(event: ProtocolEvent):
            self.finish_event(event)
            if failure_callback is not None:
                failure_callback(event.message_type)

        self.run_event(
            OutgoingTransmission(message_type, self, payload, flags),
            handle_success,
            handle_failure,
        )

    def transmit_blocking(
        self,
        message_type: MessageType,
        payload: Optional[bytes] = None,
        flags: List[TransmissionFlag] = [],
        timeout: Optional[float] = None,
        failure_callback: Optional[ProtocolFailureCallback] = None,
    ):
        is_successful, event = self.run_event_blocking(
            OutgoingTransmission(message_type, self, payload, flags), timeout
        )
        self.finish_event(event)
        if not is_successful:
            if failure_callback is not None:
                failure_callback(event.message_type)

    def request(
        self,
        message_type: MessageType,
        callback: ProtocolSuccessCallback,
        payload: Optional[bytes] = None,
        flags: List[TransmissionFlag] = [],
        retry: int = 1,
        deny_callback: Optional[ProtocolFailureCallback] = None,
    ):
        def handle_success(transmission: IncomingTransmission):
            self.finish_event(transmission)
            message_type, data, flags = transmission.get_contents()
            callback(message_type, data, flags)

        def handle_failure(event: ProtocolEvent):
            self.finish_event(event)
            if deny_callback is not None:
                deny_callback(event.message_type)

        self.run_event(
            OutgoingRequest(message_type, self, retry, payload, flags),
            handle_success,
            handle_failure,
        )

    def request_blocking(
        self,
        message_type: MessageType,
        callback: Optional[ProtocolSuccessCallback] = None,
        payload: Optional[bytes] = None,
        flags: List[TransmissionFlag] = [],
        timeout: Optional[float] = None,
        retry: int = 1,
        deny_callback: Optional[ProtocolFailureCallback] = None,
    ) -> Tuple[Optional[bytes], Optional[List[TransmissionFlag]]]:
        is_successful, event = self.run_event_blocking(
            OutgoingRequest(message_type, self, retry, payload, flags), timeout
        )
        self.finish_event(event)
        if is_successful:
            _, data, flags = event.get_contents()
            if callback is not None:
                callback(message_type, data, flags)
            return data, flags
        else:
            if deny_callback is not None:
                deny_callback(event.message_type)
            return None, None

    def listen(
        self,
        message_type: Optional[MessageType] = None,
        transmission_callback: Optional[ProtocolSuccessCallback] = None,
        request_callback: Optional[ProtocolSuccessCallback] = None,
        limit: int = -1,
        failure_callback: Optional[ProtocolFailureCallback] = None,
    ):
        listen_event = Listen(message_type, self, limit)

        def handle_success(event: ProtocolEvent):
            # We log the incoming event separately.
            KLogger.debug(f"{listen_event}: {event} has succeeded.")
            if listen_event.is_completed():
                self.finish_event(listen_event)
            message_type, data, flags = event.get_contents()
            if type(event) is IncomingRequest:
                request_callback(message_type, data, flags)
            elif type(event) is IncomingTransmission:
                transmission_callback(message_type, data, flags)
            else:
                raise ValueError(
                    "Object of invalid class passed to the success callback by"
                    f" Listen in non-blocking mode: {event}"
                )

        def handle_failure(event: ProtocolEvent):
            # We log the incoming event separately.
            KLogger.debug(f"{listen_event}: {event} has failed.")
            if listen_event.is_completed():
                self.finish_event(listen_event)
            if type(event) is IncomingTransmission:
                if failure_callback is not None:
                    failure_callback(event.message_type)
            else:
                raise ValueError(
                    "Object of invalid class passed to the failure callback by"
                    f" Listen in non-blocking mode: {event}"
                )

        self.run_event(
            listen_event,
            handle_success,
            handle_failure,
        )

    def listen_blocking(
        self,
        message_type: Optional[MessageType] = None,
        transmission_callback: Optional[ProtocolSuccessCallback] = None,
        request_callback: Optional[ProtocolSuccessCallback] = None,
        timeout: Optional[float] = None,
        failure_callback: Optional[ProtocolFailureCallback] = None,
    ) -> Tuple[
        Optional[IncomingEventType],
        Optional[MessageType],
        Optional[bytes],
        Optional[List[TransmissionFlag]],
    ]:
        is_successful, event = self.run_event_blocking(
            Listen(message_type, self, 1), timeout
        )
        self.finish_event(event)
        if is_successful:
            message_type, data, flags = event.get_contents()
            if type(event) is IncomingRequest:
                if request_callback is not None:
                    request_callback(message_type, data, flags)
                return IncomingEventType.REQUEST, message_type, data, flags
            elif type(event) is IncomingTransmission:
                if transmission_callback is not None:
                    transmission_callback(message_type, data, flags)
                return (
                    IncomingEventType.TRANSMISSION,
                    message_type,
                    data,
                    flags,
                )
            else:
                raise ValueError(
                    "Object of invalid class returned by Listen in blocking "
                    f"mode: {event}"
                )
        else:
            if failure_callback is not None:
                failure_callback(event.message_type)
            return None, None, None, None

    def event_active(self, message_type: Optional[MessageType] = None) -> bool:
        return message_type in self.current_protocol_events.keys()

    def kill_event(self, message_type: Optional[MessageType] = None):
        KLogger.debug(f"Event of type {message_type} killed.")
        with self.event_lock:
            if self.event_active(message_type):
                self.current_protocol_events.pop(message_type)
            else:
                raise ValueError(f"Event {message_type} does not exist.")
