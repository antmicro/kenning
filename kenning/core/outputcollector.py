# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides an API for processing and returning data from models and dataprovider
"""

from typing import Any, Dict, Tuple

from kenning.core.runner import Runner
from kenning.utils.args_manager import get_parsed_json_dict


class OutputCollector(Runner):

    arguments_structure = {}

    def __init__(
            self,
            inputs_sources: Dict[str, Tuple[int, str]] = {},
            inputs_specs: Dict[str, Dict] = {},
            outputs: Dict[str, str] = {}):
        """
        Creates the output collector.

        Parameters
        ----------
        inputs_sources : Dict[str, Tuple[int, str]]
            Input from where data is being retrieved
        inputs_specs : Dict[str, Dict]
            Specifications of runner's inputs
        outputs : Dict[str, str]
            Outputs of this runner
        """
        super().__init__(
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs
        )

    @classmethod
    def from_argparse(cls, args):
        """
        Constructor wrapper that takes the parameters from argparse args.

        This method takes the arguments created in form_argparse and uses them
        to create the object.

        Parameters
        ----------
        args : Dict
            arguments from ArgumentParser object

        Returns
        -------
        OutputCollector : object of class OutputCollector
        """
        return cls()

    @classmethod
    def from_json(
            cls,
            json_dict: Dict,
            inputs_sources: Dict[str, Tuple[int, str]] = {},
            inputs_specs: Dict[str, Dict] = {},
            outputs: Dict[str, str] = {}):
        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)

        return cls(
            **parsed_json_dict,
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs)

    def process_output(self, input_data: Any, output_data: Any):
        """
        Returns the infered data back to the specific place/device/connection

        Eg. it can save a video file with bounding boxes on objects or stream
        it via a TCP connection, or just show it on screen

        Parameters
        ----------
        input_data : Any
            Data collected from Datacollector that was processed by the model
        output_data : Any
            Data returned from the model
        """
        raise NotImplementedError

    def detach_from_output(self):
        """
        Detaches from the output during shutdown
        """
        raise NotImplementedError

    def should_close(self) -> bool:
        """
        Checks if a specific exit condition was reached

        This allows the OutputCollector to close gracefully if an exit
        condition was reached, eg. when a key was pressed.

        Returns
        -------
        bool : True if exit condition was reached to break the loop
        """
        raise NotImplementedError
