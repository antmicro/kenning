# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module providing a communication protocol for communication between host and
the client.
"""

from enum import Enum
from pathlib import Path
import json
import time
from typing import Any, Tuple, Optional, Union, Dict, Callable

from kenning.core.measurements import Measurements
from kenning.core.measurements import MeasurementsCollector
from kenning.utils.args_manager import ArgumentsHandler, get_parsed_json_dict
import kenning.utils.logger as logger


MSG_SIZE_LEN = 4
MSG_TYPE_LEN = 2


class RequestFailure(Exception):
    """
    Exception for failing requests.
    """
    pass


def check_request(
        request: Union[bool, Tuple[bool, Optional[bytes]]],
        msg: str):
    """
    Checks if the request finished successfully.

    When request failed, function raises RequestFailure exception.

    Parameters
    ----------
    request : Union[bool, Tuple[bool, Optional[bytes]]]
        Request result
    msg : str
        Message that should be provided with the RequestFailure exception
        when request failed.

    Raises
    ------
    RequestFailure : raised when the request did not finish successfully

    Returns
    -------
    Union[bool, Tuple[bool, Optional[bytes]]] :
        The request given in the input
    """
    if isinstance(request, bool):
        if not request:
            raise RequestFailure(f'Failed to handle request: {msg}')
    else:
        if not request[0]:
            raise RequestFailure(f'Failed to handle request: {msg}')
    return request


class MessageType(Enum):
    """
    Enum representing message type in the communication with the target device.

    For example, each message in the communication between the host and the
    target can start with 2 bytes unsigned integer representing the message
    type.

    OK - message indicating success of previous command
    ERROR - message indicating failure of previous command
    DATA - message contains inference input/output/statistics
    MODEL - message contains model to load
    PROCESS - message means the data is being processed
    OUTPUT - host requests the output from the target
    STATS - host requests the inference statistics from the target
    IOSPEC - message contains io specification to load
    """

    OK = 0
    ERROR = 1
    DATA = 2
    MODEL = 3
    PROCESS = 4
    OUTPUT = 5
    STATS = 6
    IOSPEC = 7

    def to_bytes(self, endianness: str = 'little') -> bytes:
        """
        Converts MessageType enum to bytes.

        Parameters
        ----------
        endianness : str
            Can be 'little' or 'big'

        Returns
        -------
        bytes :
            Converted message type
        """
        return int(self.value).to_bytes(MSG_TYPE_LEN, endianness, signed=False)

    @classmethod
    def from_bytes(
            cls,
            value: bytes,
            endianness: str = 'little') -> 'MessageType':
        """
        Converts bytes to MessageType enum.

        Parameters
        ----------
        value : bytes
            Enum in bytes
        endiannes : str
            Endianness in bytes

        Returns
        -------
        MessageType :
            Enum value
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
      in bytes
    * msg-type - the type of the message. For message types check the
      MessageType enum from kenning.core.runtimeprotocol
    * data - optional data that comes with the message of MessageType

    """

    def __init__(self, messsage_type: MessageType, payload: bytes = b''):
        self.message_type = messsage_type
        self.payload = payload

    @property
    def message_size(self) -> int:
        return MSG_TYPE_LEN + len(self.payload)

    @classmethod
    def from_bytes(
            cls,
            data: bytes,
            endianness: str = 'little') -> Tuple[Optional['Message'], int]:
        """
        Converts bytes to Message.

        Parameters
        ----------
        data : bytes
            Data to be converted to Message
        endianness : str
            Endianness of the bytes

        Returns
        -------
        Tuple[Optional['Message'], int] :
            Message obtained from given bytes and number of bytes used to parse
            the message
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
            data[MSG_SIZE_LEN: MSG_SIZE_LEN + MSG_TYPE_LEN],
            endianness=endianness
        )
        message_payload = data[MSG_SIZE_LEN + MSG_TYPE_LEN:][:message_size]

        return cls(message_type, message_payload), MSG_SIZE_LEN + message_size

    def to_bytes(self, endianness: str = 'little') -> bytes:
        """
        Converts Message to bytes.

        Parameters
        ----------
        endianness : str
            Endiannes of the bytes

        Returns
        -------
        bytes :
            Message converted to bytes
        """
        message_size = self.message_size
        data = message_size.to_bytes(MSG_SIZE_LEN, byteorder=endianness)
        data += self.message_type.to_bytes(endianness)
        data += self.payload

        return data

    def __eq__(self, other: 'Message') -> bool:
        if not isinstance(other, Message):
            return False
        return (self.message_type == other.message_type and
                self.payload == other.payload)

    def __repr__(self) -> str:
        return f'Message(type={self.message_type}, size={self.message_size})'


class ServerStatus(Enum):
    """
    Enum representing the status of the NetworkProtocol.serve method.

    This enum describes what happened in the last iteration of the server
    application.

    NOTHING - server reached timeout
    CLIENT_CONNECTED - new client is connected
    CLIENT_DISCONNECTED - current client is disconnected
    CLIENT_IGNORED - new client is ignored since there is already someone
        connected
    DATA_READY - data ready to process
    DATA_INVALID - data is invalid (too few bytes for the message)
    """

    NOTHING = 0
    CLIENT_CONNECTED = 1
    CLIENT_DISCONNECTED = 2
    CLIENT_IGNORED = 3
    DATA_READY = 4
    DATA_INVALID = 5


class RuntimeProtocol(ArgumentsHandler):
    """
    The interface for the communication protocol with the target devices.

    The target device acts as a server in the communication.

    The machine that runs the benchmark and collects the results is the client
    for the target device.

    The inheriting classes for this class implement at least the client-side
    of the communication with the target device.
    """

    arguments_structure = {}

    def __init__(self):
        self.log = logger.get_logger()

    @classmethod
    def from_argparse(cls, args):
        """
        Constructor wrapper that takes the parameters from argparse args.

        Parameters
        ----------
        args : Dict
            Arguments from RuntimeProtocol object

        Returns
        -------
        RuntimeProtocol :
            Object of class RuntimeProtocol
        """
        return cls()

    @classmethod
    def from_json(cls, json_dict: Dict):
        """
        Constructor wrapper that takes the parameters from json dict.

        This function checks if the given dictionary is valid according
        to the ``arguments_structure`` defined.
        If it is then it invokes the constructor.

        Parameters
        ----------
        json_dict : Dict
            Arguments for the constructor

        Returns
        -------
        RuntimeProtocol :
            Object of class RuntimeProtocol
        """

        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)

        return cls(
            **parsed_json_dict
        )

    def initialize_server(self) -> bool:
        """
        Initializes server side of the runtime protocol.

        The server side is supposed to run on target hardware.

        The parameters for the server should be provided in the constructor.

        Returns
        -------
        bool :
            True if succeded
        """
        raise NotImplementedError

    def initialize_client(self) -> bool:
        """
        Initializes client side of the runtime protocol.

        The client side is supposed to run on host testing the target hardware.

        The parameters for the client should be provided in the constructor.

        Returns
        -------
        bool :
            True if succeded
        """
        raise NotImplementedError

    def send_message(self, message: Message) -> bool:
        """
        Sends message to the target device.

        Parameters
        ----------
        message : Message
            Message to be sent

        Returns
        -------
        bool :
            True if succeeded
        """
        raise NotImplementedError

    def receive_message(
            self,
            timeout: Optional[float] = None) -> Tuple[ServerStatus, Message]:
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
        Tuple(ServerStatus, Message) :
            Tuple containing server status and received message. The status is
            NOTHING if message is incomplete and DATA_READY if it is complete
        """
        raise NotImplementedError

    def send_data(self, data: Any) -> bool:
        """
        Sends data to the target device.

        Data can be model to use, input to process, additional configuration.

        Parameters
        ----------
        data : Any
            Data to send

        Returns
        -------
        bool :
            True if successful
        """
        raise NotImplementedError

    def receive_data(
            self,
            connection: Any,
            mask: int) -> Tuple[ServerStatus, Optional[Any]]:
        """
        Receives data from the target device.

        Parameters
        ----------
        connection : Any
            Connection used to read data
        mask : int
            Selector mask from the event

        Returns
        -------
        Tuple[ServerStatus, Optional[Any]] :
            Status of receive and optionally data that was received
        """
        raise NotImplementedError

    def gather_data(
            self,
            timeout: Optional[float] = None
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
        Tuple[ServerStatus, Optional[Any]] :
            Receive status along with received data
        """
        raise NotImplementedError

    def receive_confirmation(self) -> Tuple[bool, Optional[bytes]]:
        """
        Waits until the OK message is received.

        Method waits for the OK message from the other side of connection.

        Returns
        -------
        Tuple[bool, Optional[bytes]] :
            True if OK received and attached message data, False otherwise
        """
        while True:
            status, message = self.receive_message()

            if status == ServerStatus.DATA_READY:
                if message.message_type == MessageType.ERROR:
                    self.log.error('Error during uploading input')
                    return False, None
                if message.message_type != MessageType.OK:
                    self.log.error('Unexpected message')
                    return False, None
                self.log.debug('Upload finished successfully')
                return True, message.payload

            elif status == ServerStatus.CLIENT_DISCONNECTED:
                self.log.error('Client is disconnected')
                return False, None

            elif status == ServerStatus.DATA_INVALID:
                self.log.error('Received invalid packet')
                return False, None

    def upload_input(self, data: bytes) -> bool:
        """
        Uploads input to the target device and waits for acknowledgement.

        This method should wait until the target device confirms the data is
        delivered and preprocessed for inference.

        Parameters
        ----------
        data : bytes
            Input data for inference

        Returns
        -------
        bool :
            True if ready for inference
        """
        self.log.debug('Uploading input')

        message = Message(MessageType.DATA, data)

        self.send_message(message)
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
            Path to the model

        Returns
        -------
        bool :
            True if model upload finished successfully
        """
        self.log.debug('Uploading model')
        with open(path, 'rb') as modfile:
            data = modfile.read()

        message = Message(MessageType.MODEL, data)

        self.send_message(message)
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
            Path to the json file

        Returns
        -------
        bool :
            True if data upload finished successfully
        """
        self.log.debug('Uploading io specification')
        with open(path, 'rb') as detfile:
            data = detfile.read()

        message = Message(MessageType.IOSPEC, data)

        self.send_message(message)
        return self.receive_confirmation()[0]

    def request_processing(
            self,
            get_time_func: Callable[[], float] = time.perf_counter) -> bool:
        """
        Requests processing of input data and waits for acknowledgement.

        This method triggers inference on target device and waits until the end
        of inference on target device is reached.

        This method measures processing time on the target device from the
        level of the host.

        Target may send its own measurements in the statistics.

        Parameters
        ---------
        get_time_func : Callable[[], float]
            Function that returns current timestamp

        Returns
        -------
        bool :
            True if inference finished successfully
        """
        self.log.debug('Requesting processing')
        self.send_message(Message(MessageType.PROCESS))
        start = get_time_func()
        ret = self.receive_confirmation()[0]
        if not ret:
            return False

        duration = get_time_func() - start
        measurementname = 'protocol_inference_step'
        MeasurementsCollector.measurements += {
            measurementname: [duration],
            f'{measurementname}_timestamp': [get_time_func()]
        }
        return True

    def download_output(self) -> Tuple[bool, Optional[bytes]]:
        """
        Downloads the outputs from the target device.

        Requests and downloads the latest inference output from the target
        device for quality measurements.

        Returns
        -------
        Tuple[bool, Optional[bytes]] :
            Tuple with download status (True if successful) and downloaded data
        """
        self.log.debug('Downloading output')
        self.send_message(Message(MessageType.OUTPUT))
        return self.receive_confirmation()

    def download_statistics(self) -> 'Measurements':
        """
        Downloads inference statistics from the target device.

        By default no statistics are gathered.

        Returns
        -------
        Measurements :
            Inference statistics on target device
        """
        self.log.debug('Downloading statistics')
        self.send_message(Message(MessageType.STATS))
        status, dat = self.receive_confirmation()
        measurements = Measurements()
        if status and isinstance(dat, bytes) and len(dat) > 0:
            jsonstr = dat.decode('utf8')
            jsondata = json.loads(jsonstr)
            measurements += jsondata
        return measurements

    def request_success(self, data: Optional[bytes] = bytes()) -> bool:
        """
        Sends OK message back to the client once the request is finished.

        Parameters
        ----------
        data : Optional[bytes]
            Optional data upon success, if any

        Returns
        -------
        bool :
            True if sent successfully
        """
        self.log.debug('Sending OK')

        message = Message(MessageType.OK, data)

        return self.send_message(message)

    def request_failure(self) -> bool:
        """
        Sends ERROR message back to the client if it failed to handle request.

        Returns
        -------
        bool :
            True if sent successfully
        """
        self.log.debug('Sending ERROR')

        return self.send_message(Message(MessageType.ERROR))

    def disconnect(self):
        """
        Ends connection with the other side.
        """
        raise NotImplementedError
