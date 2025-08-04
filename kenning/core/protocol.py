# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module providing a communication protocol for communication between host and
the client.
"""

import time
from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

from kenning.core.measurements import Measurements
from kenning.utils.args_manager import ArgumentsHandler
from kenning.utils.logger import KLogger


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


class ServerAction(Enum):
    """
    Enum with all types of actions/operations, that may be executed
    by Kenning inference server using the protocol.
    """

    WAITING_FOR_CLIENT = 0
    CLIENT_CONNECTED = 1
    UPLOADING_IOSPEC = 2
    UPLOADING_MODEL = 3
    UPLOADING_UNOPTIMIZED_MODEL = 4
    UPLOADING_INPUT = 5
    PROCESSING_INPUT = 6
    EXTRACTING_OUTPUT = 7
    COMPUTING_STATISTICS = 8
    OPTIMIZING_MODEL = 9
    UPLOADING_OPTIMIZERS = 10
    UPLOADING_RUNTIME = 11
    IDLE = 12

    def to_bytes(self) -> bytes:
        return self.value.to_bytes(1, "little")

    @classmethod
    def from_bytes(cls, data: bytes) -> "ServerAction":
        return cls(int.from_bytes(data, "little"))


@dataclass
class ServerStatus:
    """
    Class denoting status of an inference server (last executed action
    and information whether is succeeded or not).
    """

    last_action: ServerAction = ServerAction.WAITING_FOR_CLIENT
    success: bool = True

    def start_action(self, action: ServerAction):
        """
        Updates the status with information about new task started by the
        server. Sets the 'success' value to None (which means, that the
        action is in progress).

        Parameters
        ----------
        action: ServerAction
            Action being started.
        """
        KLogger.debug(f"Server action started: {action}")
        self.last_action = action
        self.success = None

    def finish_action(self, success: bool):
        """
        Updates the status with information about current action being
        finished and whether is succeeded or not.

        Parameters
        ----------
        success: bool
            True if action succeeded, False if not.
        """
        if success:
            KLogger.debug(f"Server action completed: {self.last_action}")
        else:
            KLogger.error(f"Server action failed: {self.last_action}")
        self.success = success


# Type definitions for 2 kinds of server callback functions (functions,
# that a server needs to implement to use the protocol) - look at the
# 'Protocol.serve' method.
ServerUploadCallback = Callable[bytes, ServerStatus]
ServerDownloadCallback = Callable[None, Tuple[ServerStatus, Optional[bytes]]]


class Protocol(ArgumentsHandler, ABC):
    """
    The interface for the communication protocol with the target devices.

    The target device acts as a server in the communication.

    The machine that runs the benchmark and collects the results is the client
    for the target device.

    The inheriting classes for this class implement at least the client-side
    of the communication with the target device.
    """

    arguments_structure = {
        "timeout": {
            "type": int,
            "description": (
                "Response receive timeout in seconds. If negative, then waits "
                "forever."
            ),
            "default": -1,
        }
    }

    def __init__(self, timeout: int = -1):
        """
        Constructs protocol.

        Parameters
        ----------
        timeout : int
            Response receive timeout in seconds. If negative, then waits for
            responses forever.
        """
        self.timeout = timeout

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
        """
        Waits for requests from the other device (the client) and calls
        appropriate callbacks. Some callbacks take bytes as arguments,
        all return ServerStatus and some also return bytes. These
        responses are then sent back to the client.

        Callbacks are guaranteed to be executed in the order requests
        were sent in.

        This method is non-blocking.

        Parameters
        ----------
        upload_input_callback: Optional[ServerUploadCallback]
            Called, when the client uploads input ('upload_input'
            method below). Should upload model input into the runtime.
        upload_model_callback: Optional[ServerUploadCallback]
            Called. when the client uploads optimized model
            ('upload_model' method below). It should load the model
            and start inference session.
        process_input_callback: Optional[ServerUploadCallback]
            Called, when the client requests inference. Should return
            after inference is completed.
        download_output_callback: Optional[ServerDownloadCallback]
            Called, when the client requests inference output, should
            return status of the server and the output.
        download_stats_callback: Optional[ServerDownloadCallback]
            Called, when the client requests inference stats, should
            end the inference session and return status of the server
            and stats.
        upload_iospec_callback: Optional[ServerUploadCallback]
            Called to upload model input/output specifications (iospec).
        upload_optimizers_callback: Optional[ServerUploadCallback]
            Called to upload optimizer config (serialized JSON)
            - 'upload_optimizers' method call by the client.
        upload_unoptimized_model_callback: Optional[ServerUploadCallback]
            Called to upload an unoptimized ML model, should save it.
        download_optimized_model_callback: Optional[ServerDownloadCallback]
            Called, when client requests optimization of the model
            (uploaded with 'upload_unoptimized_model_callback').
            Should optimize the model and return it.
        upload_runtime_callback: Optional[ServerUploadCallback]
            Called, when the client uploads runtime.
        """
        ...

    @abstractmethod
    def initialize_server(
        self,
        client_connected_callback: Optional[Callable[Any, None]] = None,
        client_disconnected_callback: Optional[Callable[None, None]] = None,
    ) -> bool:
        """
        Initializes server side of the protocol.

        The server side is supposed to run on target hardware.

        The parameters for the server should be provided in the constructor.

        Parameters
        ----------
        client_connected_callback: Optional[Callable[Any, None]]
            Called when a client connects to the server. Either IP address
            or another distinguishing characteristic of the client will be
            passed to the callback (depending on the underlying protocol).
        client_disconnected_callback: Optional[Callable[None, None]]
            Called, when the current client disconnects from the server.

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
    def upload_input(self, data: Any) -> bool:
        """
        Uploads input to the target device and waits for acknowledgement.

        This method should wait until the target device confirms the data is
        delivered and preprocessed for inference.

        Parameters
        ----------
        data : Any
            Input data for inference.

        Returns
        -------
        bool
            True if ready for inference.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def upload_runtime(self, path: Path) -> bool:
        """
        Uploads the runtime to the target device.

        This method takes the binary from given Path and sends it to the target
        device.

        This method should receive the status of runtime loading from the
        target.

        Parameters
        ----------
        path : Path
            Path to the runtime binary.

        Returns
        -------
        bool
            True if runtime upload finished successfully.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def download_output(self) -> Tuple[bool, Optional[Any]]:
        """
        Downloads the outputs from the target device.

        Requests and downloads the latest inference output from the target
        device for quality measurements.

        Returns
        -------
        Tuple[bool, Optional[Any]]
            Tuple with download status (True if successful)
            and downloaded data.
        """
        ...

    @abstractmethod
    def download_statistics(self, final: bool = False) -> Measurements:
        """
        Downloads inference statistics from the target device.

        By default no statistics are gathered.

        Parameters
        ----------
        final : bool
            If the inference is finished

        Returns
        -------
        Measurements
            Inference statistics on target device.
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    def listen_to_server_logs(self):
        """
        Starts continuously receiving and printing logs sent by the server.
        """
        ...

    @abstractmethod
    def disconnect(self):
        """
        Ends connection with the other side.
        """
        ...
