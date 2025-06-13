# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Base class for bytes-based inference communication protocol.
"""

import json
import selectors
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Tuple

from kenning.core.measurements import Measurements, timemeasurements
from kenning.core.protocol import Protocol
from kenning.utils.logger import KLogger

MSG_SIZE_LEN = 4
MSG_TYPE_LEN = 2


class MessageType(Enum):
    """
    Enum representing message type in the communication with the target device.

    For example, each message in the communication between the host and the
    target can start with 2 bytes unsigned integer representing the message
    type.

    OK - message indicating success of previous command.
    ERROR - message indicating failure of previous command.
    DATA - message contains inference input/output/statistics.
    MODEL - message contains model to load.
    PROCESS - message means the data should be processed.
    OUTPUT - host requests the output from the target.
    STATS - host requests the inference statistics from the target.
    IO_SPEC - message contains io specification to load.
    OPTIMIZERS - message contains optimizers config.
    OPTIMIZE_MODEL - message means the model should be optimized.
    RUNTIME - message contains runtime that should be used for inference
        (i.e. LLEXT binary)
    """

    OK = 0
    ERROR = 1
    DATA = 2
    MODEL = 3
    PROCESS = 4
    OUTPUT = 5
    STATS = 6
    IO_SPEC = 7
    OPTIMIZERS = 8
    OPTIMIZE_MODEL = 9
    RUNTIME = 10

    def to_bytes(
        self, endianness: Literal["little", "big"] = "little"
    ) -> bytes:
        """
        Converts MessageType enum to bytes.

        Parameters
        ----------
        endianness : Literal["little", "big"]
            Possible values are 'little' or 'big'.

        Returns
        -------
        bytes
            Converted message type.
        """
        return int(self.value).to_bytes(MSG_TYPE_LEN, endianness, signed=False)

    @classmethod
    def from_bytes(
        cls,
        value: bytes,
        endianness: Literal["little", "big"] = "little",
    ) -> "MessageType":
        """
        Converts bytes to MessageType enum.

        Parameters
        ----------
        value : bytes
            Enum in bytes.
        endianness : Literal["little", "big"]
            Endianness in bytes.

        Returns
        -------
        MessageType
            Enum value.
        """
        return MessageType(int.from_bytes(value, endianness, signed=False))


class Message(object):
    """
    Class representing single message used in protocol.

    It consists of message type and optional payload and supports conversion
    from/to bytes.

    It can be converted to byte array and has following format:

    <num-bytes><msg-type>[<data>]

    Where:

    * num-bytes - tells the size of <msg-type>[<data>] part of the message,
      in bytes.
    * msg-type - the type of the message. For message types check the
      MessageType enum from kenning.core.protocol.
    * data - optional data that comes with the message of MessageType.
    """

    def __init__(
        self, message_type: MessageType, payload: Optional[bytes] = None
    ):
        self.message_type = message_type
        self.payload = payload if payload is not None else b""

    @property
    def message_size(self) -> int:
        return MSG_TYPE_LEN + len(self.payload)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        endianness: Literal["little", "big"] = "little",
    ) -> Tuple[Optional["Message"], int]:
        """
        Converts bytes to Message.

        Parameters
        ----------
        data : bytes
            Data to be converted to Message.
        endianness : Literal["little", "big"]
            Endianness of the bytes.

        Returns
        -------
        Tuple[Optional['Message'], int]
            Message obtained from given bytes and number of bytes used to parse
            the message.
        """
        if len(data) < MSG_SIZE_LEN + MSG_TYPE_LEN:
            return None, 0

        message_size = int.from_bytes(
            data[:MSG_SIZE_LEN],
            byteorder=endianness,
        )
        if len(data) < MSG_SIZE_LEN + message_size:
            return None, 0

        message_type = MessageType.from_bytes(
            data[MSG_SIZE_LEN : MSG_SIZE_LEN + MSG_TYPE_LEN],
            endianness=endianness,
        )
        message_payload = data[MSG_SIZE_LEN + MSG_TYPE_LEN :][:message_size]

        return cls(message_type, message_payload), MSG_SIZE_LEN + message_size

    def to_bytes(
        self,
        endianness: Literal["little", "big"] = "little",
    ) -> bytes:
        """
        Converts Message to bytes.

        Parameters
        ----------
        endianness : Literal["little", "big"]
            Endianness of the bytes.

        Returns
        -------
        bytes
            Message converted to bytes.
        """
        message_size = self.message_size
        data = message_size.to_bytes(MSG_SIZE_LEN, byteorder=endianness)
        data += self.message_type.to_bytes(endianness)
        data += self.payload

        return data

    def __eq__(self, other: "Message") -> bool:
        if not isinstance(other, Message):
            return False
        return (
            self.message_type == other.message_type
            and self.payload == other.payload
        )

    def __repr__(self) -> str:
        return f"Message(type={self.message_type}, size={self.message_size})"


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

    arguments_structure = {
        "packet_size": {
            "description": "The maximum size of the received packets, in "
            "bytes.",
            "type": int,
            "default": 4096,
        },
        "endianness": {
            "description": "The endianness of data to transfer",
            "default": "little",
            "enum": ["big", "little"],
        },
    }

    def __init__(
        self,
        packet_size: int = 4096,
        endianness: str = "little",
        timeout: int = -1,
    ):
        self.packet_size = packet_size
        self.endianness = endianness
        self.selector = selectors.DefaultSelector()
        self.input_buffer = b""
        super().__init__(timeout)

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
    ) -> Tuple[ServerStatus, Message]:
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
        Tuple[ServerStatus, Message]
            Tuple containing server status and received message. The status is
            NOTHING if message is incomplete and DATA_READY if it is complete.
        """
        server_status, data = self.gather_data(timeout)
        if data is None:
            return server_status, None

        self.input_buffer += data

        message, data_parsed = Message.from_bytes(self.input_buffer)
        if message is None:
            return ServerStatus.NOTHING, None

        self.input_buffer = self.input_buffer[data_parsed:]
        KLogger.debug(f"Received message {message}")

        return ServerStatus.DATA_READY, message

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
    def receive_data(
        self, connection: Any, mask: int
    ) -> Tuple[ServerStatus, Optional[Any]]:
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
        Tuple[ServerStatus, Optional[Any]]
            Status of receive and optionally data that was received.
        """
        ...

    def gather_data(
        self, timeout: Optional[float] = None
    ) -> Tuple[ServerStatus, Optional[bytes]]:
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
        Tuple[ServerStatus, Optional[bytes]]
            Receive status along with received data.
        """
        start_time = time.perf_counter()
        while True:
            events = self.selector.select(timeout=timeout)

            results = b""
            for key, mask in events:
                if mask & selectors.EVENT_READ:
                    callback = key.data
                    server_status, data = callback(key.fileobj, mask)
                    if (
                        server_status == ServerStatus.CLIENT_DISCONNECTED
                        or data is None
                    ):
                        return server_status, None

                    results += data

            if results:
                return ServerStatus.DATA_READY, results
            elif not timeout or (time.perf_counter() - start_time > timeout):
                return ServerStatus.NOTHING, None

    def parse_message(self, message: bytes) -> Message:
        """
        Parses message from bytes.

        Parameters
        ----------
        message : bytes
            Received message.

        Returns
        -------
        Message
            Parsed message.
        """
        return Message.from_bytes(message)[0]

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
