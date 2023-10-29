# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides an API for data conversion to and from surrounding block format
during inference execution.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from kenning.utils.args_manager import ArgumentsHandler


class DataConverter(ArgumentsHandler, ABC):
    """
    Performs conversion of data between two surrounding blocks.

    This class provides an API used by Runtimes during inference execution.

    Each DataConverter should implement methods for:

    * converting data from dataset to the format used by the surrounding block.
    * converting data from format used by the surrounding block to the
    inference output.
    """

    arguments_structure: Dict[str, str] = {}

    def __init__(self):
        """
        Initializes dataprovider object.
        """
        super().__init__()

    @abstractmethod
    def to_next_block(self, data: Any) -> Any:
        """
        Converts data to the format used by the surrounding block.

        Parameters
        ----------
        data : Any
            Data to be converted.

        Returns
        -------
        Any
            Converted data.
        """
        raise NotImplementedError

    @abstractmethod
    def to_previous_block(self, data: Any) -> Any:
        """
        Converts data from the format used by the surrounding block
        to one previous block expects.

        Parameters
        ----------
        data : Any
            Data to be converted.

        Returns
        -------
        Any
            Converted data.
        """
        raise NotImplementedError
