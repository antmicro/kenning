# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides an API for data conversion to and from RuntimeProtocol format
during inference execution.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from kenning.utils.args_manager import ArgumentsHandler


class DataConverter(ArgumentsHandler, ABC):
    """
    Converts data from dataset to or from format used by the RuntimeProtocol.

    This class provides an API used by Runtimes during inference execution.

    Each DataConverter should implement methods for:

    * converting data from dataset to the format used by the RuntimeProtocol.
    * converting data from format used by the RuntimeProtocol to the inference
    output.
    """

    arguments_structure: Dict[str, str] = {}

    def __init__(self):
        """
        Initializes dataprovider object.
        """

        super().__init__()

    @abstractmethod
    def to_message(self, data: Any) -> Any:
        """
        Converts data from dataset to the format used by the RuntimeProtocol.

        Parameters
        ----------
        data : Any
            Data to be converted.

        Returns
        -------
        Any :
            Converted data.
        """
        raise NotImplementedError

    @abstractmethod
    def from_message(self, message: Any) -> Any:
        """
        Converts data from the format used by the RuntimeProtocol to the
        inference output.

        Parameters
        ----------
        message : Any
            Message to be converted.

        Returns
        -------
        Any :
            Converted data.
        """
        raise NotImplementedError
