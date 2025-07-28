# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Base class for bytes-based inference communication protocol.
"""

import json
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from kenning.core.measurements import Measurements, timemeasurements
from kenning.core.protocol import Protocol
from kenning.protocols.message import Message, MessageType
from kenning.utils.logger import KLogger


class IncomingEventType(Enum):
    """
    Enum representing types of events, that may be received by the 'listen'
    or 'listen_blocking' methods.
    """

    TRANSMISSION = 0
    REQUEST = 1


class ServerStatus(Enum):
    """
    Enum representing the status of the NetworkProtocol.serve method.

    This enum describes what happened in the last iteration of the server
    application.

    NOTHING - server reached timeout.
    CLIENT_CONNECTED - new client is connected.
    CLIENT_DISCONNECTED - current client is disconnected.
    CLIENT_IGNORED - new client is ignored since there is already someone
        connected.
    DATA_READY - data ready to process.
    DATA_INVALID - data is invalid (too few bytes for the message).
    """

    NOTHING = 0
    CLIENT_CONNECTED = 1
    CLIENT_DISCONNECTED = 2
    CLIENT_IGNORED = 3
    DATA_READY = 4
    DATA_INVALID = 5


class TransmissionFlag(Enum):
    """
    Flags, that can be send between client and server.

    Some flags are specific for 1 message type - 'message_type' property is
    storing this information ('None' means, that the flag is meant for any
    message type).


    * SUCCESS - Transmission is informing about a success.
    * FAIL - Transmission is informing about a failure. Payload is an error
      code sent by the other side.
    * IS_HOST_MESSAGE - Not set if the transmission was sent by the target
      device being evaluated. Set otherwise.
    * SERIALIZED - Flag available only for IO_SPEC transmissions. Denotes
      whether the model IO specifications are serialized.
    * IS_KENNING - Messages sent by Kenning.
    * IS_ZEPHYR - Messages sent by Kenning Zephyr Runtime.
    """

    SUCCESS = 1, None
    FAIL = 2, None
    IS_HOST_MESSAGE = 3, None
    SERIALIZED = 4, MessageType.IO_SPEC
    IS_KENNING = 5, None
    IS_ZEPHYR = 6, None

    def __new__(cls, value, message_type):
        member = object.__new__(cls)
        member._value_ = value
        member.message_type = message_type
        return member

    def for_type(self, message_type: MessageType) -> bool:
        """
        Checks, whether a given transmission flag is valid for a given
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


# Type definitions for callbacks, that are passed to lower-level methods of the
# protocol (such as request, listen, transmit).
ProtocolFailureCallback = Callable[MessageType, None]
ProtocolSuccessCallback = Callable[
    Tuple[MessageType, bytes, List[TransmissionFlag]], None
]


class BytesBasedProtocol(Protocol, ABC):
    """
    Provides methods for simple data passing, e.g. for simple
    socket-based or serial-based communication.
    """

    @abstractmethod
    def transmit(
        self,
        message_type: MessageType,
        payload: Optional[bytes] = None,
        flags: List[TransmissionFlag] = [],
        failure_callback: Optional[ProtocolFailureCallback] = None,
    ):
        """
        Sends bytes and a set of flags to the other side, without blocking
        the current thread.

        Parameters
        ----------
        message_type: MessageType
            Value denoting what is being sent. It serves to differentiate
            one transmission from another (analogous to port number in a
            TCP protocol).
        payload: Optional[bytes]
            Bytes to send (or None if the payload is to be empty).
        flags: List[TransmissionFlag]
            A list of flags to be sent (available flags in the TransmissionFlag
            enum above - please note that some flags are only allowed for a
            specific message type).
        failure_callback: Optional[ProtocolFailureCallback]
            Function, that will be called if the transmission fails to send.
            Note: This will be executed on a separate thread, so make sure the
            function is thread-safe.
        """
        ...

    @abstractmethod
    def transmit_blocking(
        self,
        message_type: MessageType,
        payload: Optional[bytes] = None,
        flags: List[TransmissionFlag] = [],
        timeout: Optional[float] = None,
        failure_callback: Optional[ProtocolFailureCallback] = None,
    ):
        """
        Sends bytes and a set of flags to the other side, blocks the current
        thread until the transmission is completed.

        Parameters
        ----------
        message_type: MessageType
            Value denoting what is being sent. It serves to differentiate
            one transmission from another (analogous to port number in a
            TCP protocol).
        payload: Optional[bytes]
            Bytes to send (or None if the payload is to be empty).
        flags: List[TransmissionFlag]
            A list of flags to be sent (available flags in the TransmissionFlag
            enum above - please note that some flags are only allowed for a
            specific message type).
        timeout: Optional[float]
            Maximum blocking time in seconds, or None to block indefinitely.
            If that time passes the 'failure_callback' will be called.
        failure_callback: Optional[ProtocolFailureCallback]
            Function, that will be called if the transmission fails to send.
        """
        ...

    @abstractmethod
    def request(
        self,
        message_type: MessageType,
        callback: ProtocolSuccessCallback,
        payload: Optional[bytes] = None,
        flags: List[TransmissionFlag] = [],
        retry: int = 1,
        deny_callback: Optional[ProtocolFailureCallback] = None,
    ):
        """
        Prompts the other side for a transmission and waits for a response,
        without blocking the current thread. Bytes and flags can also be sent
        along with the request.

        Parameters
        ----------
        message_type: MessageType
            Value denoting what type of transmission is being requested.
        callback: ProtocolSuccessCallback
            Function, that will be called when the transmission is received.
            Message type, payload and flags from the transmission will be
            passed to the function. Note: This will be executed on a separate
            thread, so make sure the callback function is thread-safe.
        payload: Optional[bytes]
            Bytes to send along with the request (or None if the payload is to
            be empty).
        flags: List[TransmissionFlag]
            A list of flags to be sent (available flags in the TransmissionFlag
            enum above - please note that some flags are only allowed for a
            specific message type).
        retry: int
            Denotes how many times the request will be re-sent after failing,
            before calling 'deny_callback'. Negative number denotes infinite
            retries.
        deny_callback: Optional[ProtocolFailureCallback]
            Function, that will be called if the request is denied or otherwise
            fails. Note: This will be executed on a separate thread, so make
            sure the callback function is thread-safe.
        """
        ...

    @abstractmethod
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
        """
        Prompts the other side for a transmission and blocks the current thread
        until a response is received. Bytes and flags can also be sent along
        with the request.

        Parameters
        ----------
        message_type: MessageType
            Value denoting what type of transmission is being requested.
        callback: Optional[ProtocolSuccessCallback]
            Function, that will be called when the transmission is received.
            Message type, payload and flags from the transmission will be
            passed to the function.
        payload: Optional[bytes]
            Bytes to send along with the request (or None if the payload is to
            be empty).
        flags: List[TransmissionFlag]
            A list of flags to be sent (available flags in the TransmissionFlag
            enum above - please note that some flags are only allowed for a
            specific message type).
        timeout: Optional[float]
            Maximum blocking time in seconds, or None to block indefinitely.
            If that time passes the 'deny_callback' will be called.
        retry: int
            Denotes how many times the request will be re-sent after failing,
            before calling 'deny_callback'. Negative number denotes infinite
            retries.
        deny_callback: Optional[ProtocolFailureCallback]
            Function, that will be called if the request is denied or otherwise
            fails.

        Returns
        -------
        Tuple[Optional[bytes], Optional[List[TransmissionFlag]]]
            Payload and flags received as response to the request, or
            (None, None) if the request was denied or otherwise failed.
        """
        ...

    @abstractmethod
    def listen(
        self,
        message_type: Optional[MessageType] = None,
        transmission_callback: Optional[ProtocolSuccessCallback] = None,
        request_callback: Optional[ProtocolSuccessCallback] = None,
        limit: int = -1,
        failure_callback: Optional[ProtocolFailureCallback] = None,
    ):
        """
        Waits for transmissions and requests from the other side, without
        blocking the current thread.

        Parameters
        ----------
        message_type: Optional[MessageType]
            Message type of the requests/transmissions to listen for, or
            None (to listen to requests/transmissions of any message type).
        transmission_callback: Optional[ProtocolSuccessCallback]
            Function, that will be called when a transmission is successfully
            received. Message type, payload and flags from the transmission
            will be passed to the function. Note: This will be executed on a
            separate thread, so make sure the callback function is thread-safe.
        request_callback: Optional[ProtocolSuccessCallback]
            Function, that will be called when a request is successfully
            received. Message type, payload and flags from the request will
            be passed to the function. Note: This will be executed on a
            separate thread, so make sure the callback function is thread-safe.
        limit: int
            Meximum number of requests/transmissions (including failures), that
            will be received before listening stops.
        failure_callback: Optional[ProtocolFailureCallback]
            Function, that will be called if a request/transmission is
            attempted by the other side, but fails. Note: This will be executed
            on a separate thread, so make sure the callback function is
            thread-safe.
        """
        ...

    @abstractmethod
    def listen_blocking(
        self,
        message_type: Optional[MessageType] = None,
        transmission_callback: Optional[ProtocolSuccessCallback] = None,
        request_callback: Optional[ProtocolSuccessCallback] = None,
        timeout: Optional[float] = None,
        failure_callback: Optional[ProtocolFailureCallback] = None,
    ) -> tuple[
        Optional[IncomingEventType],
        Optional[MessageType],
        Optional[bytes],
        Optional[List[TransmissionFlag]],
    ]:
        """
        Blocks the current thread until a transmission or a request is
        received.

        Parameters
        ----------
        message_type: Optional[MessageType]
            Message type of the requests/transmissions to listen for, or
            None (to listen to requests/transmissions of any message type).
        transmission_callback: Optional[ProtocolSuccessCallback]
            Function, that will be called when a transmission is successfully
            received. Message type, payload and flags from the transmission
            will be passed to the function.
        request_callback: Optional[ProtocolSuccessCallback]
            Function, that will be called when a request is successfully
            received. Message type, payload and flags from the request will
            be passed to the function.
        timeout: Optional[float]
            Maximum blocking time in seconds, or None to block indefinitely.
            If that time passes the 'deny_callback' will be called.
        failure_callback: Optional[ProtocolFailureCallback]
            Function, that will be called if a request/transmission is
            attempted by the other side, but fails. Note: This will be executed
            on a separate thread, so make sure the callback function is
            thread-safe.

        Returns
        -------
        tuple[Optional[IncomingEventType], Optional[MessageType], Optional[bytes], Optional[List[TransmissionFlag]]]
            Enum denoting whether a request or a transmission was received,
            message type of the received tranmsission/request, received
            payload, received flags. Alternatively: (None, None, None, None)
            if no request/transmission was received in the specified timeout
            or if a request/transmission was attempted but failed.
        """  # noqa: E501
        ...

    @abstractmethod
    def event_active(self, message_type: Optional[MessageType] = None) -> bool:
        """
        Checks if an active protocol event (Transmission, Request etc.)
        of a given message type exists.

        Parameters
        ----------
        message_type: Optional[MessageType]
            Message type to check, or None (which will check for an
            event, that accepts all message types).

        Returns
        -------
        bool
            True if event exists, False otherwise.
        """
        ...

    @abstractmethod
    def kill_event(self, message_type: Optional[MessageType] = None):
        """
        Forcibly stops an active event (Transmission, Request etc.).

        Parameters
        ----------
        message_type: Optional[MessageType]
            Message type to check, or None (which will stop an
            event, that accepts all message types).

        Raises
        ------
        ValueError
            There is no such event.
        """
        ...

    def receive_confirmation(self) -> Tuple[bool, Optional[bytes]]:
        """
        Waits until the OK message is received.

        Method waits for the OK message from the other side of connection.

        Returns
        -------
        Tuple[bool, Optional[bytes]]
            True if OK received and attached message data, False otherwise.
        """
        start_time = time.perf_counter()

        while True:
            status, message = self.receive_message(0.01)

            if status == ServerStatus.DATA_READY:
                if message.message_type == MessageType.ERROR:
                    KLogger.error("Error during uploading input")
                    return False, None
                if message.message_type != MessageType.OK:
                    KLogger.error(f"Unexpected message {message}")
                    return False, None
                KLogger.debug("Upload finished successfully")
                return True, message.payload

            elif status == ServerStatus.CLIENT_DISCONNECTED:
                KLogger.error("Client is disconnected")
                return False, None

            elif status == ServerStatus.DATA_INVALID:
                KLogger.error("Received invalid packet")
                return False, None

            if (
                self.timeout > 0
                and time.perf_counter() > start_time + self.timeout
            ):
                KLogger.error("Receive timeout")
                return False, None

    def upload_input(self, data: bytes) -> bool:
        KLogger.debug("Uploading input")

        message = Message(MessageType.DATA, data)

        if not self.send_message(message):
            return False
        return self.receive_confirmation()[0]

    def upload_model(self, path: Path) -> bool:
        KLogger.debug("Uploading model")
        with open(path, "rb") as modfile:
            data = modfile.read()

        message = Message(MessageType.MODEL, data)

        if not self.send_message(message):
            return False
        return self.receive_confirmation()[0]

    def upload_runtime(self, path: Path) -> bool:
        KLogger.debug("Uploading runtime")

        with open(path, "rb") as llext_file:
            data = llext_file.read()

        message = Message(MessageType.RUNTIME, data)

        if not self.send_message(message):
            return False
        return self.receive_confirmation()[0]

    def upload_io_specification(self, path: Path) -> bool:
        KLogger.debug("Uploading io specification")
        with open(path, "rb") as detfile:
            data = detfile.read()

        message = Message(MessageType.IO_SPEC, data)

        if not self.send_message(message):
            return False
        return self.receive_confirmation()[0]

    def request_processing(
        self, get_time_func: Callable[[], float] = time.perf_counter
    ) -> bool:
        KLogger.debug("Requesting processing")
        if not self.send_message(Message(MessageType.PROCESS)):
            return False

        ret = timemeasurements("protocol_inference_step", get_time_func)(
            self.receive_confirmation
        )()[0]
        if not ret:
            return False
        return True

    def download_output(self) -> Tuple[bool, Optional[bytes]]:
        KLogger.debug("Downloading output")
        if not self.send_message(Message(MessageType.OUTPUT)):
            return False, b""
        return self.receive_confirmation()

    def download_statistics(self, final: bool = False) -> Measurements:
        measurements = Measurements()
        if final is False:
            return measurements

        KLogger.debug("Downloading statistics")
        if not self.send_message(Message(MessageType.STATS)):
            return measurements

        status, data = self.receive_confirmation()
        if status and isinstance(data, bytes) and len(data) > 0:
            measurements += json.loads(data.decode("utf8"))
        return measurements

    def upload_optimizers(self, optimizers_cfg: Dict[str, Any]) -> bool:
        KLogger.debug("Uploading optimizers config")

        message = Message(
            MessageType.OPTIMIZERS,
            json.dumps(optimizers_cfg, default=str).encode(),
        )

        if not self.send_message(message):
            return False
        return self.receive_confirmation()[0]

    def request_optimization(
        self,
        model_path: Path,
        get_time_func: Callable[[], float] = time.perf_counter,
    ) -> Tuple[bool, Optional[bytes]]:
        KLogger.debug("Requesting model optimization")
        with open(model_path, "rb") as model_f:
            model = model_f.read()

        if not self.send_message(Message(MessageType.OPTIMIZE_MODEL, model)):
            return False, None

        ret, compiled_model_data = timemeasurements(
            "protocol_model_optimization", get_time_func
        )(self.receive_confirmation)()
        if not ret:
            return False, None
        return ret, compiled_model_data

    def request_success(self, data: Optional[bytes] = bytes()) -> bool:
        KLogger.debug("Sending OK")

        message = Message(MessageType.OK, data)

        return self.send_message(message)

    def request_failure(self) -> bool:
        KLogger.debug("Sending ERROR")

        return self.send_message(Message(MessageType.ERROR))
