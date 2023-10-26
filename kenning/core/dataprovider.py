# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides an API for gathering and preparing data from external sources.
"""

from typing import Any, Dict, Tuple
from abc import ABC, abstractmethod

from kenning.core.runner import Runner


class DataProvider(Runner, ABC):
    arguments_structure = {}

    def __init__(
        self,
        inputs_sources: Dict[str, Tuple[int, str]] = {},
        inputs_specs: Dict[str, Dict] = {},
        outputs: Dict[str, str] = {},
    ):
        """
        Initializes dataprovider object.

        Parameters
        ----------
        inputs_sources : Dict[str, Tuple[int, str]]
            Input from where data is being retrieved.
        inputs_specs : Dict[str, Dict]
            Specifications of runner's inputs.
        outputs : Dict[str, str]
            Outputs of this Runner.
        """
        self.prepare()

        super().__init__(
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs,
        )

    def prepare(self):
        """
        Prepares the source for data gathering depending on the source type.

        This will for example initialize the camera and
        set the self.device to it.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_input(self) -> Any:
        """
        Gets the sample from device.

        Returns
        -------
        Any :
            Data to be processed by the model.
        """
        raise NotImplementedError

    def preprocess_input(self, data: Any) -> Any:
        """
        Performs provider-specific preprocessing of inputs.

        Parameters
        ----------
        data : Any
            The data to be preprocessed.

        Returns
        -------
        Any :
            Preprocessed data.
        """
        return self.data

    @abstractmethod
    def detach_from_source(self):
        """
        Detaches from the source during shutdown.
        """
        raise NotImplementedError
