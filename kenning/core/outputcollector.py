# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides an API for processing and returning data from models and dataprovider.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from kenning.core.runner import Runner


class OutputCollector(Runner, ABC):
    """
    Collects outputs from models running in the Kenning flow.

    It performs final processing of data running in the Kenning
    flow.
    It can be used i.e. to display predictions, save them to file
    or send to other application.
    """

    arguments_structure = {}

    def __init__(
        self,
        inputs_sources: Dict[str, Tuple[int, str]] = {},
        inputs_specs: Dict[str, Dict] = {},
        outputs: Dict[str, str] = {},
    ):
        """
        Creates the output collector.

        Parameters
        ----------
        inputs_sources : Dict[str, Tuple[int, str]]
            Input from where data is being retrieved.
        inputs_specs : Dict[str, Dict]
            Specifications of runner's inputs.
        outputs : Dict[str, str]
            Outputs of this runner.
        """
        super().__init__(
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs,
        )

    @abstractmethod
    def process_output(self, input_data: Any, output_data: Any):
        """
        Returns the inferred data back to the specific place/device/connection.

        Eg. it can save a video file with bounding boxes on objects or stream
        it via a TCP connection, or just show it on screen.

        Parameters
        ----------
        input_data : Any
            Data collected from Datacollector that was processed by the model.
        output_data : Any
            Data returned from the model.
        """
        ...

    @abstractmethod
    def detach_from_output(self):
        """
        Detaches from the output during shutdown.
        """
        ...

    @abstractmethod
    def should_close(self) -> bool:
        """
        Checks if a specific exit condition was reached.

        This allows the OutputCollector to close gracefully if an exit
        condition was reached, eg. when a key was pressed.

        Returns
        -------
        bool
            True if exit condition was reached to break the loop.
        """
        ...
