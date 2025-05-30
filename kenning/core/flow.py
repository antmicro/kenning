# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
Provides implementation of a KenningFlow that allows execution of arbitrary
flows created from Runners.
"""

from typing import Any, Dict, List

import jsonschema

from kenning.core.runner import Runner
from kenning.utils.class_loader import load_class
from kenning.utils.logger import KLogger


class KenningFlow:
    """
    Allows for creation of custom flows using Kenning core classes.

    KenningFlow class creates and executes customized flows consisting of
    the runners implemented based on kenning.core classes, such as
    DatasetProvider, ModelRunner, OutputCollector.
    Designed flows may be formed into non-linear, graph-like structures.

    The flow may be defined either directly via dictionaries or in a predefined
    JSON format.

    The JSON format must follow well defined structure.
    Each runner should consist of following entries:

    type - Type of a Kenning class to use for this module
    parameters - Inner parameters of chosen class
    inputs - Optional, set of pairs (local name, global name)
    outputs - Optional, set of pairs (local name, global name)

    All global names (inputs and outputs) must be unique.
    All local names are predefined for each class.
    All variables used as input to a runner must be defined as a output of a
    runner that is placed before that runner.
    """

    def __init__(self, runners: List[Runner]):
        """
        Initializes the flow.

        Parameters
        ----------
        runners : List[Runner]
            List of specifications of runners in the flow.
        """
        self.runners = runners
        self.flow_state = None
        self.should_close = False

    @classmethod
    def form_parameterschema(cls) -> Dict:
        """
        Creates schema for the KenningFlow class.

        Returns
        -------
        Dict
            Schema for the class.
        """
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "parameters": {"type": "object"},
                    "inputs": {
                        "type": "object",
                        "patternProperties": {".": {"type": "string"}},
                    },
                    "outputs": {
                        "type": "object",
                        "patternProperties": {".": {"type": "string"}},
                    },
                    "additionalProperties": False,
                },
                "required": ["type", "parameters"],
            },
        }

    @classmethod
    def from_json(
        cls, runners_specifications: List[Dict[str, Any]]
    ) -> "KenningFlow":
        """
        Constructor wrapper that takes the parameters from json dict.

        This function checks if the given dictionary is valid according
        to the json schema defined in ``form_parameterschema``.
        If it is then it parses json and invokes the constructor.

        Parameters
        ----------
        runners_specifications : List[Dict[str, Any]]
            List of runners that creates the flow.

        Returns
        -------
        KenningFlow
            Object of class KenningFlow.

        Raises
        ------
        jsonschema.ValidationError
            Raised for invalid JSON description
        Exception
            Raised for undefined and redefined variables, depending on context
        """
        try:
            jsonschema.validate(
                runners_specifications, cls.form_parameterschema()
            )
        except jsonschema.ValidationError as e:
            KLogger.error(f"JSON description is invalid: {e.message}")
            raise

        output_variables = {}
        output_specs = {}
        runners: List[Runner] = []

        # order blocks to properly create variables
        runners_idx_ordered = []

        previous_len = 0
        missing_vars = []

        found_vars = []
        while len(runners_idx_ordered) < len(runners_specifications):
            for runner_idx, runner_spec in enumerate(runners_specifications):
                if runner_idx in runners_idx_ordered:
                    continue
                inputs = runner_spec.get("inputs", {})
                outputs = runner_spec.get("outputs", {})
                missing_vars_runner = [
                    var for var in inputs.values() if var not in found_vars
                ]
                if len(missing_vars_runner) > 0:
                    missing_vars.extend(missing_vars_runner)
                    continue
                runners_idx_ordered.append(runner_idx)
                for global_name in outputs.values():
                    # leaving redefinition check to graph forming part
                    if global_name not in found_vars:
                        found_vars.append(global_name)
            if previous_len == len(runners_idx_ordered):
                raise Exception(
                    f"Scenario has undeclared input variables:  {set(missing_vars)}"  # noqa: E501
                )
            previous_len = len(runners_idx_ordered)
            missing_vars = []

        # graph forming
        for runner_idx in runners_idx_ordered:
            runner_spec = runners_specifications[runner_idx]
            try:
                runner_cls: Runner = load_class(runner_spec["type"])
                cfg = runner_spec["parameters"]
                inputs = runner_spec.get("inputs", {})
                outputs = runner_spec.get("outputs", {})

                KLogger.info(f"Loading runner: {runner_cls.__name__}")

                # validate output variables and add them to dict
                for local_name, global_name in outputs.items():
                    if global_name in output_variables.keys():
                        KLogger.error(
                            f"Error loading runner {runner_idx}: {runner_cls}"
                        )
                        KLogger.error(
                            f"Redefined output variable {local_name}:"
                            f"{global_name}"
                        )
                        raise Exception(
                            f"Redefined output variable {global_name}"
                        )

                    output_variables[global_name] = runner_idx

                # create and fill dict with input sources
                inputs_sources = {}

                for local_name, global_name in inputs.items():
                    if global_name not in output_variables:
                        KLogger.error(
                            f"Error loading runner {runner_idx}:"
                            f"{runner_cls}. Undefined input variable "
                            f"{local_name}:{global_name}."
                        )
                        raise Exception(
                            f"Undefined input variable {global_name}"
                        )

                    inputs_sources[local_name] = (
                        output_variables[global_name],
                        global_name,
                    )

                # get output specs from global flow variables
                inputs_specs = {}
                for local_name, (_, global_name) in inputs_sources.items():
                    inputs_specs[local_name] = output_specs[global_name]

                # instantiate runner
                runner = runner_cls.from_json(
                    cfg,
                    inputs_sources=inputs_sources,
                    inputs_specs=inputs_specs,
                    outputs=outputs,
                )

                # populate dict with flow variables specs
                runner_io_spec = runner.get_io_specification()
                runner_output_specification = runner_io_spec[
                    "output"
                ] + runner_io_spec.get("processed_output", [])

                for local_name, global_name in runner.outputs.items():
                    for out_spec in runner_output_specification:
                        if out_spec["name"] == local_name:
                            output_specs[global_name] = out_spec

                runners.append(runner)

            except Exception as e:
                KLogger.error(f"Error during flow json parsing: {e}")
                for runner in runners:
                    runner.cleanup()
                raise

        return cls(runners)

    def init_state(self):
        self.flow_state = []

    def cleanup(self):
        for runner in self.runners:
            runner.cleanup()

    def run_single_step(self):
        """
        Runs flow one time.
        """
        try:
            for runner in self.runners:
                if runner.should_close():
                    self.should_close = True
                    break

                runner._run(self.flow_state)
        except Exception:
            self.cleanup()
            raise

    def run(self):
        """
        Main process function. Repeatedly runs constructed graph in a loop.
        """
        while not self.should_close:
            self.init_state()
            try:
                self.run_single_step()

            except KeyboardInterrupt:
                KLogger.warning(
                    "Processing interrupted due to keyboard interrupt"
                )
                KLogger.warning("Aborting")
                break

            except StopIteration:
                KLogger.warning(
                    "Processing interrupted due to an empty stream"
                )
                break

            except NotImplementedError:
                KLogger.error("Missing implementation of action from module")
                break

            except RuntimeError as e:
                KLogger.error("Processing interrupted from inside of module")
                KLogger.error(e)
                break

        self.cleanup()

        KLogger.debug(f"Final {self.flow_state=}")
