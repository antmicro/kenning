# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
Provides implementation of a KenningFlow that allows execution of arbitrary
flows created from Runners.
"""

from typing import Dict, Any, List
import jsonschema

from kenning.interfaces.io_interface import IOInterface
from kenning.interfaces.io_interface import IOCompatibilityError
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
        runners: List[Runner] = []

        for runner_idx, runner_spec in enumerate(runners_specifications):
            runner_cls: Runner = load_class(runner_spec['type'])
            cfg = runner_spec['parameters']
            inputs = runner_spec.get('inputs', {})
            outputs = runner_spec.get('outputs', {})

            log.info(f'Loading runner: {runner_cls.__name__}')

            # validate output variables and add them to dict
            for local_name, global_name in outputs.items():
                if global_name in output_variables.keys():
                    log.error(f'Error loading runner {runner_idx}:'
                              f'{runner_cls}. Redefined output variable '
                              f'{local_name}:{global_name}.')
                    raise Exception(f'Redefined output variable {global_name}')

                output_variables[global_name] = runner_idx

            # create and fill dict with input sources
            inputs_sources = {}

            for local_name, global_name in inputs.items():
                if global_name not in output_variables:
                    log.error(f'Error loading runner {runner_idx}:'
                              f'{runner_cls}. Undefined input variable '
                              f'{local_name}:{global_name}.')
                    raise Exception(f'Undefined input variable {global_name}')

                inputs_sources[local_name] = (output_variables[global_name],
                                              global_name)

            # instantiate runner
            runner = runner_cls.from_json(
                cfg,
                inputs_sources=inputs_sources,
                outputs=outputs
            )

            runners.append(runner)

        cls._validate_runners_io(runners)

        return cls(runners)

    @staticmethod
    def _validate_runners_io(runners: List[Runner]):
        """
        Validates IO of runners. If there is some incompatibility then an
        Exceptions is raised.

        Parameters
        ----------
        runners : List[Runner]
            List of runners that creates the flow
        """

        output_specifications = {}

        for runner in runners:
            # populate dict with flow variables specs
            runner_output_specification = \
                (runner.get_io_specification()['output']
                 + runner.get_io_specification().get('processed_output', []))

            for local_name, global_name in runner.outputs.items():
                for out_spec in runner_output_specification:
                    if out_spec['name'] == local_name:
                        output_specifications[global_name] = out_spec

        for runner in runners:
            # get output specs from global flow variables
            output_spec = {}
            for _, (_, name) in runner.inputs_sources.items():
                output_spec[name] = output_specifications[name]

            # get input specs mapped to global variables
            input_spec = {}
            runner_io_spec = runner.get_io_specification()
            for local_name, (_, global_name) in runner.inputs_sources.items():
                for spec in runner_io_spec['input']:
                    if spec['name'] == local_name:
                        input_spec[global_name] = spec
                        break

            if not IOInterface.validate(output_spec, input_spec):
                raise IOCompatibilityError(
                    f'Input and output are not compatible.\nOutput is:\n'
                    f'{output_spec}\nInput is:\n{input_spec}\n'
                )

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
            self.flow_state = []
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

        for runner in self.runners:
            runner.cleanup()

        self.log.info(f'Final {self.flow_state=}')
