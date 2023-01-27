# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module providing a communication protocol for communication between host and
the client.
"""

from enum import Enum
from pathlib import Path
import argparse

from typing import Any, Tuple, List, Optional, Union, Dict

from kenning.core.measurements import Measurements
from kenning.utils.args_manager import get_parsed_json_dict, add_argparse_argument, add_parameterschema_argument  # noqa: E501


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
    Union[bool, Tuple[bool, Optional[bytes]]] : the request given in the input
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

    def to_bytes(self, endianness: str = 'little') -> str:
        """
        Converts MessageType enum to bytes in uint16 format.

        Parameters
        ----------
        endianness : str
            Can be 'little' or 'big'

        Returns
        -------
        bytes : converted message type
        """
        return int(self.value).to_bytes(2, endianness, signed=False)

    @classmethod
    def from_bytes(
            cls,
            value: bytes,
            endianness: str = 'little') -> 'MessageType':
        """
        Converts 2-byte bytes to MessageType enum.

        Parameters
        ----------
        value : bytes
            enum in bytes
        endiannes : str
            endianness in bytes

        Returns
        -------
        MessageType : enum value
        """
        return MessageType(int.from_bytes(value, endianness, signed=False))


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


class RuntimeProtocol(object):
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
        pass

    @classmethod
    def _form_argparse(cls):
        """
        Wrapper for creating argparse structure for the RuntimeProtocol class.

        Returns
        -------
        (ArgumentParser, ArgumentGroup) :
            tuple with the argument parser object that can act as parent for
            program's argument parser, and the corresponding arguments' group
            pointer
        """
        parser = argparse.ArgumentParser(
            add_help=False,
            conflict_handler='resolve'
        )
        group = parser.add_argument_group(title='Runtime protocol arguments')
        return parser, group

    @classmethod
    def form_argparse(cls):
        """
        Creates argparse parser for the RuntimeProtocol object.

        Returns
        -------
        (ArgumentParser, ArgumentGroup) :
            tuple with the argument parser object that can act as parent for
            program's argument parser, and the corresponding arguments' group
            pointer
        """
        parser, group = cls._form_argparse()
        if cls.arguments_structure != RuntimeProtocol.arguments_structure:
            add_argparse_argument(
                group,
                cls.arguments_structure
            )
        return parser, group

    @classmethod
    def from_argparse(cls, args):
        """
        Constructor wrapper that takes the parameters from argparse args.

        Parameters
        ----------
        args : Dict
            arguments from RuntimeProtocol object

        Returns
        -------
        RuntimeProtocol : object of class RuntimeProtocol
        """
        return cls()

    @classmethod
    def _form_parameterschema(cls):
        """
        Wrapper for creating parameterschema structure
        for the RuntimeProtocol class.

        Returns
        -------
        Dict : schema for the class
        """
        parameterschema = {
            "type": "object",
            "additionalProperties": False
        }

        return parameterschema

    @classmethod
    def form_parameterschema(cls):
        """
        Creates schema for the RuntimeProtocol class.

        Returns
        -------
        Dict : schema for the class
        """
        parameterschema = cls._form_parameterschema()
        if cls.arguments_structure != RuntimeProtocol.arguments_structure:
            add_parameterschema_argument(
                parameterschema,
                cls.arguments_structure
            )
        return parameterschema

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
        RuntimeProtocol : object of class RuntimeProtocol
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
        bool : True if succeded
        """
        raise NotImplementedError

    def initialize_client(self) -> bool:
        """
        Initializes client side of the runtime protocol.

        The client side is supposed to run on host testing the target hardware.

        The parameters for the client should be provided in the constructor.

        Returns
        -------
        bool : True if succeded
        """
        raise NotImplementedError

    def wait_for_activity(self) -> List[Tuple['ServerStatus', Any]]:
        """
        Waits for incoming data from the other side of connection.

        This method should wait for the input data to arrive and return the
        appropriate status code along with received data.

        Returns
        -------
        List[Tuple['ServerStatus', Any]] :
            list of messages along with status codes.
        """
        raise NotImplementedError

    def send_data(self, data: bytes) -> bool:
        """
        Sends data to the target device.

        Data can be model to use, input to process, additional configuration.

        Parameters
        ----------
        data : bytes
            Data to send

        Returns
        -------
        bool : True if successful
        """
        raise NotImplementedError

    def receive_data(self) -> Tuple['ServerStatus', Any]:
        """
        Gathers data from the client.

        This method should be called by wait_for_activity method in order to
        receive data from the client.

        Returns
        -------
        Tuple[ServerStatus, Any] : receive status along with received data
        """
        raise NotImplementedError

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
        bool : True if ready for inference
        """
        raise NotImplementedError

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
        bool : True if model upload finished successfully
        """
        raise NotImplementedError

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
        bool : True if data upload finished successfully
        """
        raise NotImplementedError

    def request_processing(self) -> bool:
        """
        Requests processing of input data and waits for acknowledgement.

        This method triggers inference on target device and waits until the end
        of inference on target device is reached.

        This method measures processing time on the target device from the
        level of the host.

        Target may send its own measurements in the statistics.

        Returns
        -------
        bool : True if inference finished successfully
        """
        raise NotImplementedError

    def download_output(self) -> Tuple[bool, Optional[bytes]]:
        """
        Downloads the outputs from the target device.

        Requests and downloads the latest inference output from the target
        device for quality measurements.

        Returns
        -------
        Tuple[bool, Optional[bytes]] : tuple with download status (True if
            successful) and downloaded data
        """
        raise NotImplementedError

    def download_statistics(self) -> 'Measurements':
        """
        Downloads inference statistics from the target device.

        By default no statistics are gathered.

        Returns
        -------
        Measurements : inference statistics on target device
        """
        return Measurements()

    def request_success(self, data: bytes = bytes()) -> bool:
        """
        Sends OK message back to the client once the request is finished.

        Parameters
        ----------
        data : bytes
            Optional data upon success, if any

        Returns
        -------
        bool : True if sent successfully
        """
        raise NotImplementedError

    def request_failure(self) -> bool:
        """
        Sends ERROR message back to the client if it failed to handle request.

        Returns
        -------
        bool : True if sent successfully
        """
        raise NotImplementedError

    def parse_message(self, message: bytes) -> Tuple['MessageType', bytes]:
        """
        Parses message received in the wait_for_activity method.

        The message type is determined from its contents and the optional data
        is returned along with it.

        Parameters
        ----------
        message : bytes
            Received message

        Returns
        -------
        Tuple['MessageType', bytes] : message type and accompanying data
        """
        raise NotImplementedError

    def disconnect(self):
        """
        Ends connection with the other side.
        """
        raise NotImplementedError
