# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module providing a communication protocol for communication between host and
the client.
"""

import json
import time
from abc import ABC, abstractmethod
from argparse import Namespace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

from kenning.core.measurements import Measurements, timemeasurements
from kenning.utils.args_manager import ArgumentsHandler
from kenning.utils.logger import KLogger

MSG_SIZE_LEN = 4
MSG_TYPE_LEN = 2

ZEPHYR_LLEXT_NAME_MAX_LEN = 32


class RequestFailure(Exception):
    """
    Exception for failing requests.
    """

    pass


def check_request(
    request: Union[bool, Tuple[bool, Optional[bytes]]], msg: str
) -> Tuple[bool, Optional[bytes]]:
    """
    Checks if the request finished successfully.

    When request failed, function raises RequestFailure exception.

    Parameters
    ----------
    request : Union[bool, Tuple[bool, Optional[bytes]]]
        Request result.
    msg : str
        Message that should be provided with the RequestFailure exception
        when request failed.

    Returns
    -------
    Tuple[bool, Optional[bytes]]
        The request given in the input.

    Raises
    ------
    RequestFailure :
        Raised when the request did not finish successfully.
    """
    if isinstance(request, bool):
        request = request, None
    if not request[0]:
        raise RequestFailure(f"Failed to handle request: {msg}")

    return request


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
    LOAD_LLEXT - message contains compiled Zephyr's LLEXT.
    UNLOAD_LLEXT - hosts requests unloading of the given LLEXT.
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
    ZEPHYR_LOAD_LLEXT = 10
    ZEPHYR_UNLOAD_LLEXT = 11

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


class Protocol(ArgumentsHandler, ABC):
    """
    The interface for the communication protocol with the target devices.

    The target device acts as a server in the communication.

    The machine that runs the benchmark and collects the results is the client
    for the target device.

    The inheriting classes for this class implement at least the client-side
    of the communication with the target device.
    """

    arguments_structure = {}

    @classmethod
    def from_argparse(cls, args: Namespace) -> "Protocol":
        """
        Constructor wrapper that takes the parameters from argparse args.

        Parameters
        ----------
        args : Namespace
            Arguments from Protocol object.

        Returns
        -------
        Protocol
            Object of class Protocol.
        """
        return super().from_argparse(args)

    @classmethod
    def from_json(cls, json_dict: Dict) -> "Protocol":
        """
        Constructor wrapper that takes the parameters from json dict.

        This function checks if the given dictionary is valid according
        to the ``arguments_structure`` defined.
        If it is then it invokes the constructor.

        Parameters
        ----------
        json_dict : Dict
            Arguments for the constructor.

        Returns
        -------
        Protocol
            Object of class Protocol.
        """
        return super().from_json(json_dict)

    @abstractmethod
    def initialize_server(self) -> bool:
        """
        Initializes server side of the protocol.

        The server side is supposed to run on target hardware.

        The parameters for the server should be provided in the constructor.

        Returns
        -------
        bool
            True if succeeded.
        """
        ...

    @abstractmethod
    def initialize_client(self) -> bool:
        """
        Initializes client side of the protocol.

        The client side is supposed to run on host testing the target hardware.

        The parameters for the client should be provided in the constructor.

        Returns
        -------
        bool
            True if succeeded.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

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

    @abstractmethod
    def gather_data(
        self, timeout: Optional[float] = None
    ) -> Tuple[ServerStatus, Optional[Any]]:
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
        Tuple[ServerStatus, Optional[Any]]
            Receive status along with received data.
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

    def upload_input(self, data: bytes) -> bool:
        """
        Uploads input to the target device and waits for acknowledgement.

        This method should wait until the target device confirms the data is
        delivered and preprocessed for inference.

        Parameters
        ----------
        data : bytes
            Input data for inference.

        Returns
        -------
        bool
            True if ready for inference.
        """
        KLogger.debug("Uploading input")

        message = Message(MessageType.DATA, data)

        if not self.send_message(message):
            return False
        return self.receive_confirmation()[0]

    def upload_model(self, path: Path) -> bool:
        """
        Uploads the model to the target device.

        This method takes the model from given Path and sends it to the target
        device.

        This method should receive the status of uploading the model from the
        target.

        Parameters
        ----------
        path : Path
            Path to the model.

        Returns
        -------
        bool
            True if model upload finished successfully.
        """
        KLogger.debug("Uploading model")
        with open(path, "rb") as modfile:
            data = modfile.read()

        message = Message(MessageType.MODEL, data)

        if not self.send_message(message):
            return False
        return self.receive_confirmation()[0]

    def zephyr_load_llext(self, name: str, path: Path) -> bool:
        """
        Uploads the loadable linkable extension to the target device.

        This method takes the binary from given Path and sends it to the target
        device.

        This method should receive the status of loading LLEXT from the
        target.

        Parameters
        ----------
        name : str
            Name of the extension.
        path : Path
            Path to the LLEXT binary.

        Returns
        -------
        bool
            True if LLEXT upload finished successfully.
        """
        KLogger.debug(f"Loading LLEXT {name}")

        data = (
            name[: ZEPHYR_LLEXT_NAME_MAX_LEN - 1]
            .encode()
            .ljust(ZEPHYR_LLEXT_NAME_MAX_LEN, b"\0")
        )

        with open(path, "rb") as llext_file:
            data += llext_file.read()

        message = Message(MessageType.ZEPHYR_LOAD_LLEXT, data)

        if not self.send_message(message):
            return False
        return self.receive_confirmation()[0]

    def zephyr_unload_llext(self, name: str) -> bool:
        """
        Requests unloading of the given extension.

        This method should receive the status of unloading LLEXT from the
        target.

        Parameters
        ----------
        name : str
            Name of the extension.

        Returns
        -------
        bool
            True if LLEXT upload finished successfully.
        """
        KLogger.debug(f"Unloading LLEXT {name}")

        data = (
            name[:ZEPHYR_LLEXT_NAME_MAX_LEN]
            .encode()
            .ljust(ZEPHYR_LLEXT_NAME_MAX_LEN, b"\0")
        )

        message = Message(MessageType.ZEPHYR_UNLOAD_LLEXT, data)

        if not self.send_message(message):
            return False
        return self.receive_confirmation()[0]

    def upload_io_specification(self, path: Path) -> bool:
        """
        Uploads input/output specification to the target device.

        This method takes the specification in a json format from the given
        Path and sends it to the target device.

        This method should receive the status of uploading the data to
        the target.

        Parameters
        ----------
        path : Path
            Path to the json file.

        Returns
        -------
        bool
            True if data upload finished successfully.
        """
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
        """
        Requests processing of input data and waits for acknowledgement.

        This method triggers inference on target device and waits until the end
        of inference on target device is reached.

        This method measures processing time on the target device from the
        level of the host.

        Target may send its own measurements in the statistics.

        Parameters
        ----------
        get_time_func : Callable[[], float]
            Function that returns current timestamp.

        Returns
        -------
        bool
            True if inference finished successfully.
        """
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
        """
        Downloads the outputs from the target device.

        Requests and downloads the latest inference output from the target
        device for quality measurements.

        Returns
        -------
        Tuple[bool, Optional[bytes]]
            Tuple with download status (True if successful)
            and downloaded data.
        """
        KLogger.debug("Downloading output")
        if not self.send_message(Message(MessageType.OUTPUT)):
            return False, b""
        return self.receive_confirmation()

    def download_statistics(self) -> Measurements:
        """
        Downloads inference statistics from the target device.

        By default no statistics are gathered.

        Returns
        -------
        Measurements
            Inference statistics on target device.
        """
        measurements = Measurements()

        KLogger.debug("Downloading statistics")
        if not self.send_message(Message(MessageType.STATS)):
            return measurements

        status, data = self.receive_confirmation()
        if status and isinstance(data, bytes) and len(data) > 0:
            measurements += json.loads(data.decode("utf8"))
        return measurements

    def upload_optimizers(self, optimizers_cfg: Dict[str, Any]) -> bool:
        """
        Upload optimizers config to the target device.

        Parameters
        ----------
        optimizers_cfg : Dict[str, Any]
            Config JSON of optimizers.

        Returns
        -------
        bool
            True if data upload finished successfully.
        """
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
        """
        Request optimization of model.

        Parameters
        ----------
        model_path : Path
            Path to the model for optimization.
        get_time_func : Callable[[], float]
            Function that returns current timestamp.

        Returns
        -------
        Tuple[bool, Optional[bytes]]
            First element is equal to True if optimization finished
            successfully and the second element contains compiled model.
        """
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
        """
        Sends OK message back to the client once the request is finished.

        Parameters
        ----------
        data : Optional[bytes]
            Optional data upon success, if any.

        Returns
        -------
        bool
            True if sent successfully.
        """
        KLogger.debug("Sending OK")

        message = Message(MessageType.OK, data)

        return self.send_message(message)

    def request_failure(self) -> bool:
        """
        Sends ERROR message back to the client if it failed to handle request.

        Returns
        -------
        bool
            True if sent successfully.
        """
        KLogger.debug("Sending ERROR")

        return self.send_message(Message(MessageType.ERROR))

    @abstractmethod
    def disconnect(self):
        """
        Ends connection with the other side.
        """
        ...
