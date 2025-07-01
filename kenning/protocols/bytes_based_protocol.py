# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
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
from math import ceil
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
)

from kenning.core.measurements import Measurements, timemeasurements
from kenning.core.protocol import Protocol
from kenning.utils.logger import KLogger
from kenning.utils.serializable import Serializable

"""
Below constants denote sizes of fields in a serialized message.
"""

# Size (in bytes and bits) of the 'message_size' field
MSG_SIZE_LEN = 4
MSG_SIZE_BITS = MSG_SIZE_LEN * 8

# Size in bytes of the combined fields 'message_type' and 'flow_control_flags'.
MSG_ID_SIZE = 1

# Size in bits of the 'message_type' field.
MSG_TYPE_BITS = 6

# Size in bits of the 'flow_control_flags' field.
FLOW_CONTROL_FLAG_COUNT = 2

# Size in bytes of the 'checksum' field.
CHECKSUM_SIZE = 1

# Number of flags in the 'general_purpose_flags' field (including reserved
# flags) and thus size in bits of that field.
GENERAL_FLAG_COUNT = 12

# Number of flags in the 'message_type_specific_flags' field (including
# reserved flags) and thus size in bits of that field.
MSG_TYPE_SPECIFIC_FLAG_COUNT = 4

# Total number of flags and size in bits of the 'flags' field.
FLAG_COUNT = GENERAL_FLAG_COUNT + MSG_TYPE_SPECIFIC_FLAG_COUNT

# Size in bytes of the 'flags' field.
FLAGS_SIZE = ceil(FLAG_COUNT / 8)

# Size of the entire message header.
HEADER_SIZE = MSG_ID_SIZE + CHECKSUM_SIZE + FLAGS_SIZE + MSG_SIZE_LEN


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

    def __int__(self):
        return self.value


class FlowControlFlags(Enum):
    """
    Enum representing all possible states in flow control of the protocol.

    * REQUEST - Requesting a transmission of a given Message Type. Should be
      responded to with a transmission, or with a negative acknowledgement (if
      the request is denied).
    * REQUEST_RETRANSMIT - Message requesting a retransmission due to an error.
      Size field of the message should be set to the number of message that is
      to be retransmitted. Setting the field to 0xFFFFFFFF means that the whole
      transmission is to be retried. Should be responded to with a
      transmission, or with a negative acknowledgement (if the request is
      denied)
    * ACKNOWLEDGE - Message sent to acknowledge receiving a transmission, a
      request or another acknowledgement. Depending on whether FAIL or SUCCESS
      flag is set be a negative response (rejecting a request, a transmission
      or an acknowledgement) or a positive response (informing, that the
      transmission was received successfully). When acknowledging transmission
      the size field of the message should be set to the number of received
      messages.
    * TRANSMISSION - Message carrying payload.
    """

    REQUEST = 0b00
    REQUEST_RETRANSMIT = 0b01
    ACKNOWLEDGE = 0b10
    TRANSMISSION = 0b11

    def __int__(self):
        return self.value


class MessageID(Serializable):
    """
    Enum representing Message Identifier - comprised of Message Type and Flow
    Control Flags. Message type denotes what is being sent/requested (i.e.
    model output, data, iospec), while Flow Control Flags denote whether it's a
    request, response to a request etc.
    """

    serializable_fields = (
        ("message_type", MessageType, MSG_TYPE_BITS),
        ("flow_control_flags", FlowControlFlags, FLOW_CONTROL_FLAG_COUNT),
    )

    def __init__(
        self,
        message_type: MessageType = 0,
        flow_control_flags: FlowControlFlags = 0,
    ):
        """
        Initializes the object. Default values are 0.

        Parameters
        ----------
        message_type : MessageType
            Type of the message.

        flow_control_flags: FlowControlFlags
            Value decoding function of the message in the flow of the protocol.
        """
        super().__init__()
        self.message_type = message_type
        self.flow_control_flags = flow_control_flags


class FlagName(Enum):
    """
    Enum with all flag names allowed in messages.

    * SUCCESS - Message is informing about a success (of the operation running
      on the other device, like inference).
    * FAIL - Message is informing about a failure on the other side.
    * IS_HOST_MESSAGE - Set to 0 if message was sent from the target device
      being evaluated (1 otherwise).
    * HAS_PAYLOAD - Payload field is present in the message.
    * FIRST - First message in the transmission.
    * LAST - Last message in the transmission.
    * SPEC_FLAG_1..SPEC_FLAG_4 - Flags specific for a message type.
    """

    SUCCESS = "success"
    FAIL = "fail"
    IS_HOST_MESSAGE = "is_host_message"
    HAS_PAYLOAD = "has_payload"
    FIRST = "first"
    LAST = "last"
    SPEC_FLAG_1 = "spec_flag_1"
    SPEC_FLAG_2 = "spec_flag_2"
    SPEC_FLAG_3 = "spec_flag_3"
    SPEC_FLAG_4 = "spec_flag_4"

    def __str__(self):
        return self.value


class Flags(Serializable):
    """
    Class representing a 2-byte flag field of a message. Youngest 12 bits are
    general purpose flags, oldest 4 bits are message type specific flags.

    Flags format, youngest bit on the left:
    SFIHRA``````1234

    * ` - reserved
    * S - SUCCESS
    * F - FAIL
    * I - IS_HOST_MESSAGE
    * H - HAS_PAYLOAD
    * R - FIRST
    * A - LAST
    * 1..4 - SPEC_FLAG_1..SPEC_FLAG_4
    """

    # Flags available
    serializable_fields = (
        (FlagName.SUCCESS, bool, 1),
        (FlagName.FAIL, bool, 1),
        (FlagName.IS_HOST_MESSAGE, bool, 1),
        (FlagName.HAS_PAYLOAD, bool, 1),
        (FlagName.FIRST, bool, 1),
        (FlagName.LAST, bool, 1),
        ("reserved", bool, 6),
        (FlagName.SPEC_FLAG_1, bool, 1),
        (FlagName.SPEC_FLAG_2, bool, 1),
        (FlagName.SPEC_FLAG_3, bool, 1),
        (FlagName.SPEC_FLAG_4, bool, 1),
    )

    def __init__(
        self,
        flags: Dict[FlagName, bool] = {},
    ):
        """
        Initializes the flags. By default all flags are set to 0, unless
        overridden by the Dict passed as argument.

        Parameters
        ----------
        flags : Dict[FlagName, bool]
            Flags, that will be set to specified values.
        """
        super().__init__()
        for flag, value in flags.items():
            setattr(self, str(flag), value)


"""
Type returned by the Message.from_bytes method, containing results of
attempted de-serialization.

* message - Message class object, deserialized from bytes, or None if bytes did
  not contain a full message (because it wasn't fully received yet).
* bytes_parsed - Number of bytes used to parse the message.
* checksum_valid - True, if the the parsed message's checksum was correct,
  False it was not correct, or None if bytes did not contain a full message.
"""
DeserializedMessage = NamedTuple(
    "DeserializedMessage",
    [
        ("message", Optional[ForwardRef("Message")]),
        ("bytes_parsed", int),
        ("checksum_valid", Optional[bool]),
    ],
)


class Message(object):
    """
    Class representing single message used in protocol.

    It can be converted to byte array and has following format:

    <msg-id><checksum><flags><payload_size>[<payload>]

    Where:

    * msg-id - 8 bit value, youngest 6 bits are message type, oldest 2 bits
      are flow control flags. For message types check the MessageType enum.
      For flow control values check FlowControlFlags enum.
    * checksum - 8 bit value, XOR of all other bytes in the message
    * flags - 16 bit value, youngest 12 bits are general flags, oldest 4
      bits are flags specific for a given massage type. See Flags class and
      FlagName enum for details
    * payload_size - size of the payload
    * payload - optional data that comes with the message.
    """

    # Value, that all bytes in the message will be XOR-ed with, to compute
    # the checksum.
    CHECKSUM_MAGIC = 0x4B

    def __init__(
        self,
        message_type: MessageType,
        payload: Optional["bytes"] = None,
        flow_control_flags: FlowControlFlags = FlowControlFlags.TRANSMISSION,
        flags: Flags = Flags(),
    ):
        """
        Initializes the object.

        Parameters
        ----------
        message_type : MessageType
            Type of the Message.
        payload : Optional["bytes"]
            A stream of bytes to send.
        flow_control_flags : FlowControlFlags
            Value decoding function of the message in the flow of the protocol.
        flags : Flags
            A set of flags sent with the message.

        Raises
        ------
        ValueError
            Message payload was given, but is not of type 'bytes'.
        """
        self.message_type = message_type
        self.flow_control_flags = flow_control_flags
        self.flags = flags
        if payload is not None and type(payload) is not bytes:
            raise ValueError(
                f"Invalid payload: {payload} (payload must be serialized)."
            )
        self.message_size = len(payload) if payload is not None else 0
        self.payload = payload if payload is not None else b""
        flags.has_payload = 1 if self.message_size > 0 else 0

    @staticmethod
    def _xor_bytes(data: bytes) -> int:
        """
        Computes XOR of all bytes in a bytestream.

        Parameters
        ----------
        data : bytes
            Data to process.

        Returns
        -------
        int
            Computed XOR - an 8 bit value.
        """
        result = 0
        for byte in data:
            result ^= byte
        return result

    def _compute_checksum(self) -> int:
        """
        Computes parity byte checksum, which is a XOR of all bytes in the
        message, except the checksum byte, with a special key value.

        Returns
        -------
        int
            Computed checksum - an 8 bit value.
        """
        data = self.to_bytes(True)
        return self.CHECKSUM_MAGIC ^ self._xor_bytes(data)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        endianness: Literal["little", "big"] = "little",
    ) -> DeserializedMessage:
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
        DeserializedMessage
            Tuple containing: Message parsed from given bytes, number of bytes
            used to parse the message and information whether the message
            checksum is valid.

            If the bytes do not contain a whole message (because the message
            hasn't been fully received yet), value: (None, 0, None) will be
            returned.
        """
        if len(data) < HEADER_SIZE:
            return DeserializedMessage(
                message=None, bytes_parsed=0, checksum_valid=None
            )
        message_id = MessageID.from_bytes(data[:MSG_ID_SIZE])
        message_type = message_id.message_type
        flow_control_flags = message_id.flow_control_flags
        flags = Flags.from_bytes(
            data[
                MSG_ID_SIZE + CHECKSUM_SIZE : MSG_ID_SIZE
                + CHECKSUM_SIZE
                + FLAGS_SIZE
            ],
        )
        payload_size = int.from_bytes(
            data[
                MSG_ID_SIZE + CHECKSUM_SIZE + FLAGS_SIZE : MSG_ID_SIZE
                + CHECKSUM_SIZE
                + FLAGS_SIZE
                + MSG_SIZE_LEN
            ],
            byteorder=endianness,
        )
        if flags.has_payload:
            full_message_size = payload_size + HEADER_SIZE
        else:
            full_message_size = HEADER_SIZE
        if len(data) < full_message_size:
            return DeserializedMessage(
                message=None, bytes_parsed=0, checksum_valid=None
            )
        if flags.has_payload:
            message_payload = data[HEADER_SIZE:full_message_size]
        else:
            message_payload = b""
        checksum_valid = (
            cls._xor_bytes(data[:full_message_size]) == cls.CHECKSUM_MAGIC
        )
        return DeserializedMessage(
            message=cls(
                message_type, message_payload, flow_control_flags, flags
            ),
            bytes_parsed=full_message_size,
            checksum_valid=checksum_valid,
        )

    def to_bytes(
        self,
        set_checksum_to_zero: bool = False,
        endianness: Literal["little", "big"] = "little",
    ) -> bytes:
        """
        Converts Message to bytes.

        Parameters
        ----------
        set_checksum_to_zero: bool
            When set to True, the method will place a 0 in the checksum field,
            instead of actually computing the checksum. It is needed, because
            the 'compute_checksum' method uses this method.
        endianness : Literal["little", "big"]
            Endianness of the bytes.

        Returns
        -------
        bytes
            Message converted to bytes.
        """
        message_id = MessageID(self.message_type, self.flow_control_flags)
        if set_checksum_to_zero:
            checksum = 0
        else:
            checksum = self._compute_checksum()
        data = message_id.to_bytes()
        data += checksum.to_bytes(CHECKSUM_SIZE, byteorder=endianness)
        data += self.flags.to_bytes()
        data += self.message_size.to_bytes(MSG_SIZE_LEN, byteorder=endianness)
        data += self.payload
        return data

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

        message, data_parsed, checksum_valid = Message.from_bytes(
            self.input_buffer
        )
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
