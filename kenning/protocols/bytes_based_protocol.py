# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Base class for bytes-based inference communication protocol.
"""

import json
import logging
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
from kenning.core.protocol import (
    Protocol,
    ServerAction,
    ServerDownloadCallback,
    ServerStatus,
    ServerUploadCallback,
)
from kenning.protocols.message import (
    MessageType,
)
from kenning.utils.logger import KLogger

KLogger.add_custom_level(logging.INFO + 2, "DEVICE")


class IncomingEventType(Enum):
    """
    Enum representing types of events, that may be received by the 'listen'
    or 'listen_blocking' methods.
    """

    TRANSMISSION = 0
    REQUEST = 1


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
    Implements abstract methods from the Protocol class, using an underlying
    mechanism of transmissions and requests (that mechanism needs to be
    provided by this class'es extensions).
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

    def check_status(
        self,
        status: ServerStatus,
        artifacts: Tuple[bytes, List[TransmissionFlag]],
    ) -> bool:
        """
        Parses ServerStatus from the contents of a transmission (payload and
        flags) and compares it against given status.

        Parameters
        ----------
        status: ServerStatus
            Status to compare against.
        artifacts: Tuple[bytes, List[TransmissionFlag]]
            Payload and bytes from a transmission. Flags should either
            contain 'IS_ZEPHYR' flag or the 'IS_KENNING' flag.

        Returns
        -------
        bool
            True if status is equal, False if status is not as expected, or
            if some or the 'artifacts' are None.

        Raises
        ------
        ValueError
            Flags do not containing 'IS_ZEPHYR' flag or the 'IS_KENNING' flag.
        """
        payload, flags = artifacts
        if payload is None or flags is None:
            return False
        elif TransmissionFlag.IS_KENNING in flags:
            received_status = ServerStatus(
                ServerAction.from_bytes(payload),
                True
                if TransmissionFlag.SUCCESS in flags
                else False
                if TransmissionFlag.FAIL in flags
                else None,
            )
            return status == received_status
        elif TransmissionFlag.IS_ZEPHYR in flags:
            return TransmissionFlag.SUCCESS in flags
        else:
            raise ValueError("Received status from an unknown source.")

    def upload_input(self, data: bytes) -> bool:
        KLogger.debug("Uploading input")
        return self.check_status(
            ServerStatus(ServerAction.UPLOADING_INPUT),
            self.request_blocking(MessageType.DATA, None, data),
        )

    def upload_model(self, path: Path) -> bool:
        KLogger.debug("Uploading model")
        with open(path, "rb") as modfile:
            data = modfile.read()
        return self.check_status(
            ServerStatus(ServerAction.UPLOADING_MODEL),
            self.request_blocking(MessageType.MODEL, None, data),
        )

    def upload_runtime(self, path: Path) -> bool:
        KLogger.debug("Uploading runtime")

        with open(path, "rb") as llext_file:
            data = llext_file.read()
        return self.check_status(
            ServerStatus(ServerAction.UPLOADING_RUNTIME),
            self.request_blocking(MessageType.RUNTIME, None, data),
        )

    def upload_io_specification(self, path: Path) -> bool:
        KLogger.debug("Uploading io specification")
        with open(path, "rb") as detfile:
            data = detfile.read()
        return self.check_status(
            ServerStatus(ServerAction.UPLOADING_IOSPEC),
            self.request_blocking(MessageType.IO_SPEC, None, data),
        )

    def request_processing(
        self, get_time_func: Callable[[], float] = time.perf_counter
    ) -> bool:
        KLogger.debug("Requesting processing")
        ret = timemeasurements("protocol_inference_step", get_time_func)(
            self.request_blocking
        )(MessageType.PROCESS)
        return self.check_status(
            ServerStatus(ServerAction.PROCESSING_INPUT), ret
        )

    def download_output(self) -> Tuple[bool, Optional[bytes]]:
        KLogger.debug("Downloading output")
        output, flags = self.request_blocking(MessageType.OUTPUT)
        return output is not None and TransmissionFlag.SUCCESS in flags, output

    def download_statistics(self, final: bool = False) -> Measurements:
        measurements = Measurements()
        if final is False:
            return measurements

        KLogger.debug("Downloading statistics")

        data, flags = self.request_blocking(MessageType.STATS)
        if (
            TransmissionFlag.SUCCESS in flags
            and isinstance(data, bytes)
            and len(data) > 0
        ):
            measurements += json.loads(data.decode("utf8"))
        return measurements

    def upload_optimizers(self, optimizers_cfg: Dict[str, Any]) -> bool:
        KLogger.debug("Uploading optimizers config")
        return self.check_status(
            ServerStatus(ServerAction.UPLOADING_OPTIMIZERS),
            self.request_blocking(
                MessageType.OPTIMIZERS,
                None,
                json.dumps(optimizers_cfg, default=str).encode(),
            ),
        )

    def request_optimization(
        self,
        model_path: Path,
        get_time_func: Callable[[], float] = time.perf_counter,
    ) -> Tuple[bool, Optional[bytes]]:
        KLogger.debug("Requesting model optimization")
        with open(model_path, "rb") as model_f:
            model = model_f.read()
        self.request_blocking(MessageType.UNOPTIMIZED_MODEL, None, model)
        compiled_model_data, flags = timemeasurements(
            "protocol_model_optimization", get_time_func
        )(self.request_blocking)(MessageType.OPTIMIZE_MODEL)
        return (
            (
                compiled_model_data is not None
                and TransmissionFlag.SUCCESS in flags
            ),
            compiled_model_data,
        )

    def listen_to_server_logs(self):
        def parse_logs(
            message_type: MessageType,
            data: bytes,
            flags: List[TransmissionFlag],
        ):
            if TransmissionFlag.IS_ZEPHYR in flags:
                while len(data) > 0:
                    size = data[0]
                    KLogger.device(data[1:size].decode("ascii"))
                    data = data[size:]
            elif TransmissionFlag.IS_KENNING in flags:
                raise NotImplementedError
            else:
                KLogger.warning("Received logs from unknown source.")

        KLogger.debug("Receiving logs from the server...")
        self.listen(MessageType.LOGS, parse_logs)

    def serve(
        self,
        upload_input_callback: Optional[ServerUploadCallback] = None,
        upload_model_callback: Optional[ServerUploadCallback] = None,
        process_input_callback: Optional[ServerUploadCallback] = None,
        download_output_callback: Optional[ServerDownloadCallback] = None,
        download_stats_callback: Optional[ServerDownloadCallback] = None,
        upload_iospec_callback: Optional[ServerUploadCallback] = None,
        upload_optimizers_callback: Optional[ServerUploadCallback] = None,
        upload_unoptimized_model_callback: Optional[
            ServerUploadCallback
        ] = None,
        download_optimized_model_callback: Optional[
            ServerDownloadCallback
        ] = None,
        upload_runtime_callback: Optional[ServerUploadCallback] = None,
    ):
        self.server_upload_callbacks = {
            MessageType.DATA: upload_input_callback,
            MessageType.MODEL: upload_model_callback,
            MessageType.PROCESS: process_input_callback,
            MessageType.IO_SPEC: upload_iospec_callback,
            MessageType.OPTIMIZERS: upload_optimizers_callback,
            MessageType.UNOPTIMIZED_MODEL: upload_unoptimized_model_callback,
            MessageType.RUNTIME: upload_runtime_callback,
        }

        self.server_download_callbacks = {
            MessageType.OPTIMIZE_MODEL: download_optimized_model_callback,
            MessageType.STATS: download_stats_callback,
            MessageType.OUTPUT: download_output_callback,
        }

        def handle_request(
            message_type: MessageType,
            payload: bytes,
            flags: List[TransmissionFlag],
        ):
            if (
                message_type in self.server_download_callbacks
                and self.server_download_callbacks[message_type] is not None
            ):
                status, data = self.server_download_callbacks[message_type]()
                if status.success:
                    self.transmit(
                        message_type,
                        data,
                        [
                            TransmissionFlag.SUCCESS,
                            TransmissionFlag.IS_KENNING,
                        ],
                    )
                else:
                    self.transmit(
                        message_type,
                        status.last_action.to_bytes(),
                        [TransmissionFlag.FAIL, TransmissionFlag.IS_KENNING],
                    )
            elif (
                message_type in self.server_upload_callbacks
                and self.server_upload_callbacks[message_type] is not None
            ):
                status = self.server_upload_callbacks[message_type](payload)
                self.transmit(
                    message_type,
                    status.last_action.to_bytes(),
                    [
                        TransmissionFlag.IS_KENNING,
                        TransmissionFlag.SUCCESS
                        if status.success
                        else TransmissionFlag.FAIL,
                    ],
                )

        self.listen(None, None, handle_request)
