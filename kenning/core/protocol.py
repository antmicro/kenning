# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module providing a communication protocol for communication between host and
the client.
"""

import time
from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

from kenning.core.measurements import Measurements
from kenning.utils.args_manager import ArgumentsHandler


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

    @abstractmethod
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
        ...

    @abstractmethod
    def request_failure(self) -> bool:
        """
        Sends ERROR message back to the client if it failed to handle request.

        Returns
        -------
        bool
            True if sent successfully.
        """
        ...

    @abstractmethod
    def disconnect(self):
        """
        Ends connection with the other side.
        """
        ...
