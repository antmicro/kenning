from typing import Any

import jsonschema
from kenning.utils import logger

from kenning.utils.class_loader import load_class


class KenningFlow:
    """
    Allows for creation of custom flows using Kenning core classes.

    KenningFlow class creates and executes customized flows consisting of
    the modules implemented based on kenning.core classes, such as
    Dataset, ModelWrapper, Runtime.
    Designed flows may be formed into non-linear, graph-like structures.

    The flow may be defined either directly via dictionaries or in a predefined
    JSON format.

    The JSON format must follow well defined structure.
    Each module should be prefixed with a unique name and consist of
    following entires:

    type - Type of a Kenning class to use for this module
    parameters - Inner parameters of chosen class
    inputs - Optional, set of pairs (local name, global name)
    outputs - Optional, set of pairs (local name, global name)
    action - Singular string denoting intended action for a module

    All global names (inputs and outputs) must be unique.
    All local names are predefined for each class.
    """

    def __init__(
            self,
            modules: dict[str, tuple[type, Any, str]],
            inputs: dict[str, dict[str, str]],
            outputs: dict[str, dict[str, str]]):
        """
        Initializes the flow. This constructor only invokes helper functions.

        Parameters
        ----------
        modules : Dict
            Mapping of module names to their types, config and action
        inputs : Dict
            Mapping of module names to (global input name -> local input name)
        outputs : Dict
            Mapping of module names to (local input name -> global input name)
        """

        self.log = logger.get_logger()
        self.modules = dict()
        self.inputs = inputs
        self.outputs = outputs

        self._create_flow(modules)

        if self._has_cycles():
            raise RuntimeError('Resulting graph has possible cycles')

        # TODO implement compile function body
        # self.compile()

    def _create_flow(self, modules: dict[str, tuple[type, Any, str]]):
        """
        This helper function creates the flow. It will call the constructors
        of underlaying classes.

        Parameters
        ----------
            modules : Dict
                Mapping of module names to their types, config and action
        """

        for name, (type, cfg, action) in modules.items():
            try:
                self.modules[name] = (type.from_json(cfg), action)

            # TODO remove this after removing dataset from arguments
            except TypeError:
                ds_name = self._find_input_module(name)
                self.modules[name] = (type.from_json(
                    self.modules[ds_name][0], cfg), action)

            except Exception as e:
                self.log.error(f'Error loading submodule {name} : {str(e)}')
                raise

    def _find_input_module(self, name: str) -> str:
        """
        Helper function returning name of a input module for given node.

        Parameters
        ----------
        name : str
            Name of module we are searching parent for

        Returns
        -------
        str : Name of a module that provides input for given node
        """
        try:
            return [
                module_name for module_name in self.modules if
                set(self.inputs[name]) &
                set(self.outputs[module_name].values()) != set()
            ][0]
        except IndexError:
            raise IndexError('Module input connector not found')

    def _depth_first_search(
            self,
            matrix: list[list[int]],
            visited: list[bool],
            node: int) -> bool:
        """
        Depth first search helper function
        Parameters
        ----------
        matrix : Adjacency matrix
        visited : Local list noting visited nodes
        node : Current node

        Returns
        -------
            bool : Wheher a cycle was found
        """
        if visited[node]:
            return True

        visited[node] = True

        for n, conn in enumerate(matrix[node]):
            if conn:
                if self._depth_first_search(matrix, visited, n):
                    return True

        visited[node] = False
        return False

    def _has_cycles(self) -> bool:
        """
        Helper function that checks for possible cycles in a graph.
        Possible cycles are deduced only from static description of
        inputs and outputs. This means that not every possible cycle
        has to actually occur during the processing. An example would be
        a situation, where action defined within module does not yield
        an output that would close the cycle inside the graph.
        That also implies such defined connection would be redundant
        and should be removed from graph description.

        Returns
        -------
            bool : Whether graph has possible cycles
        """
        matrix = [[0 for _ in self.modules] for _ in self.modules]

        for n1, m1 in enumerate(self.modules):
            for n2, m2 in enumerate(self.modules):
                if m1 in self.outputs and m2 in self.inputs:
                    s1 = self.outputs[m1].values()
                    s2 = self.inputs[m2].keys()
                    matrix[n1][n2] = len(s1 & s2)

        for node in range(len(self.modules)):
            if self._depth_first_search(
                    matrix, [False for _ in self.modules],
                    node):
                return True

        return False

    def compile(self):
        """
        This function runs sequential optimizaions on models and
        prepares all runtimes for deployment.
        """
        raise NotImplementedError('Compile function not yet implemented')

    @classmethod
    def form_parameterschema(cls):
        """
        Creates schema for the KenningFlow class

        Returns
        -------
            Dict : Schema for the class
        """
        return {
            'type': 'object',
            'patternProperties': {
                '.': {
                    'type': 'object',
                    'properties': {
                        'type': {'type': 'string'},
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
                        'properties': {
                            'type': 'object'
                        },
                        'additionalProperties': False
                    },
                    'required': ['type', 'parameters']
                }
            }
        }

    @classmethod
    def from_json(cls, json_dict: dict[str, Any]):
        log = logger.get_logger()

        try:
            jsonschema.validate(json_dict, cls.form_parameterschema())
        except jsonschema.ValidationError:
            log.error('JSON description is invalid')
            raise

        modules = dict()
        inputs = dict()
        outputs = dict()

        for module_name, module_cfg in json_dict.items():
            modules[module_name] = (
                load_class(module_cfg['type']),
                module_cfg['parameters'],
                module_cfg['action'])
            try:
                inputs[module_name] = {
                    global_name: local_name for local_name,
                    global_name in module_cfg['inputs'].items()}
            except KeyError:
                pass

            try:
                outputs[module_name] = {
                    local_name: global_name for local_name,
                    global_name in module_cfg['outputs'].items()}
            except KeyError:
                pass

        return cls(modules, inputs, outputs)

    def step(self):
        current_outputs = dict()

        for name, (module, action) in self.modules.items():
            input = {
                local_name: current_outputs[global_name]
                for global_name, local_name in
                self.inputs[name].items()
            } if name in self.inputs else {}

            output = module.actions[action](input)

            current_outputs.update({
                self.outputs[name][local_output]: value
                for local_output, value in
                output.items()
            })

        return current_outputs

    def process(self):
        """
        Main process function. Repeatedly runs constructed graph in a loop.
        """
        current_outputs = dict()
        while True:
            try:
                current_outputs = self.step()

            except KeyboardInterrupt:
                self.log.warn('Processing interrupted due to keyboard interrupt. Aborting.')  # noqa: E501
                break

            except StopIteration:
                self.log.warn('Processing interrupted due to an empty stream.')
                break

            except NotImplementedError:
                self.log.error('Missing implementation of action from module')
                break

            except RuntimeError as e:
                self.log.warn(f'Processing interrupted from inside of module. {(str(e))}')  # noqa: E501
                break

        return current_outputs
