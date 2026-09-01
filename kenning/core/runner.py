# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a base class for Kenning Flow elements.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from kenning.interfaces.io_interface import (
    IOInterface,
    ModulesIncompatibleError,
)
from kenning.utils.args_manager import ArgumentsHandler


class Runner(IOInterface, ArgumentsHandler, ABC):
    """
    Represents an operation block in Kenning Flow.
    """

    def __init__(
        self,
        inputs_sources: Dict[str, Tuple[int, str]],
        inputs_specs: Dict[str, Dict],
        outputs: Dict[str, str],
    ):
        """
        Creates the runner.

        Parameters
        ----------
        inputs_sources : Dict[str, Tuple[int, str]]
            Input from where data is being retrieved.
        inputs_specs : Dict[str, Dict]
            Specifications of runner's inputs.
        outputs : Dict[str, str]
            Outputs of this Runner.

        Raises
        ------
        ModulesIncompatibleError
            Raised when connections have incompatible types
        """
        self.inputs_sources = inputs_sources
        self.inputs_specs = inputs_specs
        self.outputs = outputs

        # get input specs mapped to global variables
        runner_input_spec = {}
        runner_io_spec = self.get_io_specification()

        found_global_mapping = False
        for specname in ("input", "processed_input"):
            if specname not in runner_io_spec:
                continue
            for spec in runner_io_spec[specname]:
                if found_global_mapping:
                    break
                for local_name, (
                    _,
                    global_name,
                ) in self.inputs_sources.items():
                    if spec["name"] == local_name:
                        runner_input_spec[global_name] = (
                            [spec] if isinstance(spec, Dict) else spec
                        )
                        found_global_mapping = True
                        break

        if (
            not found_global_mapping
            and runner_io_spec["input"]
            and inputs_sources
        ):
            raise ModulesIncompatibleError(
                "io_specification is incompatible with "
                "inputs specified in configuration"
            )

        # get provided inputs spec mapped to global variables
        outputs_specs = {}
        for local_name, (_, global_name) in self.inputs_sources.items():
            outputs_specs[global_name] = self.inputs_specs[local_name]
            if not isinstance(outputs_specs[global_name], List):
                outputs_specs[global_name] = [outputs_specs[global_name]]

        if not IOInterface.validate(outputs_specs, runner_input_spec):
            self.cleanup()
            raise ModulesIncompatibleError(
                f"Input and output are not compatible.\nOutput is:\n"
                f"{outputs_specs}\nInput is:\n{runner_input_spec}\n"
            )

    def cleanup(self):
        """
        Method that cleans resources after Runner is no longer needed.
        """
        pass

    def should_close(self) -> bool:
        """
        Method that checks if Runner got some exit indication (exception etc.)
        and the flow should close.

        Returns
        -------
        bool
            True if there was some exit indication.
        """
        return False

    def _run(self, flow_state: List[Dict[str, Any]]):
        """
        Method used to prepare inputs and run this Runner.

        Parameters
        ----------
        flow_state : List[Dict[str, Any]]
            Current flow state containing all variables used in flow.
        """
        # retrieves input values from current flow state based on data
        # saved in input sources (block index and block output name)
        inputs = {
            input_name: flow_state[block_idx][output_name]
            for input_name, (
                block_idx,
                output_name,
            ) in self.inputs_sources.items()
        }
        local_outputs = self.run(inputs)
        outputs = {}
        for local_name, global_name in self.outputs.items():
            outputs[global_name] = local_outputs[local_name]

        if outputs:
            flow_state.append(outputs)

    @abstractmethod
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Method used to run this Runner.

        Parameters
        ----------
        inputs : Dict[str, Any]
            Inputs provided to this block.

        Returns
        -------
        Dict[str, Any]
            Output of this block.
        """
        ...
