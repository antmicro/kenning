# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Base class for bytes-based inference communication protocol.
"""

import json
import time
from abc import ABC
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
)

from kenning.core.measurements import Measurements, timemeasurements
from kenning.core.protocol import Protocol
from kenning.protocols.message import Message, MessageType
from kenning.utils.logger import KLogger


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


class BytesBasedProtocol(Protocol, ABC):
    """
    Provides methods for simple data passing, e.g. for simple
    socket-based or serial-based communication.
    """

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
