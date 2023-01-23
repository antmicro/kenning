# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
Provides implementation of a KenningFlow that allows execution of arbitrary
flows created from Runners.
"""

from typing import Dict, Any, List
import jsonschema

from kenning.core.runner import Runner
from kenning.utils import logger
from kenning.utils.class_loader import load_class


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
    Each runner should consist of following entires:

    type - Type of a Kenning class to use for this module
    parameters - Inner parameters of chosen class
    inputs - Optional, set of pairs (local name, global name)
    outputs - Optional, set of pairs (local name, global name)

    All global names (inputs and outputs) must be unique.
    All local names are predefined for each class.
    All variables used as input to a runner must be defined as a output of a
    runner that is placed before that runner.
    """

    def __init__(
            self,
            runners: List[Runner]):
        """
        Initializes the flow.

        Parameters
        ----------
        runners_specifications : List[Runner]
            List of specifications of runners in the flow
        """

        self.log = logger.get_logger()

        self.runners = runners
        self.flow_state = None
        self.should_close = False

    @classmethod
    def form_parameterschema(cls):
        """
        Creates schema for the KenningFlow class

        Returns
        -------
        Dict :
            Schema for the class
        """
        return {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'type': {
                        'type': 'string'
                    },
                    'parameters': {
                        'type': 'object'
                    },
                    'inputs': {
                        'type': 'object',
                        'patternProperties': {
                            '.': {'type': 'string'}
                        }
                    },
                    'outputs': {
                        'type': 'object',
                        'patternProperties': {
                            '.': {'type': 'string'}
                        }
                    },
                    'additionalProperties': False
                },
                'required': ['type', 'parameters']
            }
        }

    @classmethod
    def from_json(cls, runners_specifications: List[Dict[str, Any]]):
        """
        Constructor wrapper that takes the parameters from json dict.

        This function checks if the given dictionary is valid according
        to the json schema defined in ``form_parameterschema``.
        If it is then it parses json and invokes the constructor.

        Parameters
        ----------
        runners_specifications : List
            List of runners that creates the flow.

        Returns
        -------
        KenningFlow :
            object of class KenningFlow
        """
        log = logger.get_logger()

        try:
            jsonschema.validate(
                runners_specifications,
                cls.form_parameterschema())
        except jsonschema.ValidationError as e:
            log.error(f'JSON description is invalid: {e.message}')
            raise

        output_variables = {}
        output_specs = {}
        runners: List[Runner] = []

        for runner_idx, runner_spec in enumerate(runners_specifications):
            try:
                runner_cls: Runner = load_class(runner_spec['type'])
                cfg = runner_spec['parameters']
                inputs = runner_spec.get('inputs', {})
                outputs = runner_spec.get('outputs', {})

                log.info(f'Loading runner: {runner_cls.__name__}')

                # validate output variables and add them to dict
                for local_name, global_name in outputs.items():
                    if global_name in output_variables.keys():
                        log.error(
                            f'Error loading runner {runner_idx}:'
                            f'{runner_cls}. Redefined output variable '
                            f'{local_name}:{global_name}.'
                        )
                        raise Exception(
                            f'Redefined output variable {global_name}'
                         )

                    output_variables[global_name] = runner_idx

                # create and fill dict with input sources
                inputs_sources = {}

                for local_name, global_name in inputs.items():
                    if global_name not in output_variables:
                        log.error(
                            f'Error loading runner {runner_idx}:'
                            f'{runner_cls}. Undefined input variable '
                            f'{local_name}:{global_name}.'
                        )
                        raise Exception(
                            f'Undefined input variable {global_name}'
                        )

                    inputs_sources[local_name] = (
                        output_variables[global_name],
                        global_name
                    )

                # get output specs from global flow variables
                inputs_specs = {}
                for local_name, (_, glbal_name) in inputs_sources.items():
                    inputs_specs[local_name] = output_specs[glbal_name]

                # instantiate runner
                runner = runner_cls.from_json(
                    cfg,
                    inputs_sources=inputs_sources,
                    inputs_specs=inputs_specs,
                    outputs=outputs
                )

                # populate dict with flow variables specs
                runner_io_spec = runner.get_io_specification()
                runner_output_specification = \
                    (runner_io_spec['output']
                     + runner_io_spec.get('processed_output', []))

                for local_name, global_name in runner.outputs.items():
                    for out_spec in runner_output_specification:
                        if out_spec['name'] == local_name:
                            output_specs[global_name] = out_spec

                runners.append(runner)

            except Exception as e:
                log.error(f'Error during flow json parsing: {e}')
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
        self.log.info('Flow started')
        for runner in self.runners:
            if runner.should_close():
                self.should_close = True
                break

            runner._run(self.flow_state)

    def run(self):
        """
        Main process function. Repeatedly runs constructed graph in a loop.
        """

        while not self.should_close:
            self.init_state()
            try:
                self.run_single_step()

            except KeyboardInterrupt:
                self.log.warn('Processing interrupted due to keyboard '
                              'interrupt. Aborting.')
                break

            except StopIteration:
                self.log.warn('Processing interrupted due to an empty stream.')
                break

            except NotImplementedError:
                self.log.error('Missing implementation of action from module')
                break

            except RuntimeError as e:
                self.log.warn('Processing interrupted from inside of module. '
                              f'{(str(e))}')
                break

        self.cleanup()

        self.log.info(f'Final {self.flow_state=}')
