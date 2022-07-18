from typing import Any, Dict, Tuple, Type

import jsonschema
from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.utils import logger

from kenning.utils.class_loader import load_class


class KenningFlow:
    """
    Allows for creation of custom flows using Kenning core classes.

    KenningFlow class allowes for creation and execution of customized flows,
    utilizing modules available in the framework (i.e. Dataset, ModelWrapper).
    Designed flows may be formed into non-linear, graph-like structures.

    The flow may be defined either directly via dictionaries or in a predefined
    JSON format.
    """

    def __init__(self, modules: Dict[str, Tuple[Type, Any, str]],
                 inputs: Dict[str, Dict[str, str]],
                 outputs: Dict[str, Dict[str, str]]):
        """
        Creates and compiles a flow.

        Parameters
        ----------
        modules : Dict
            Mapping of module names to their types, config and action
        inputs : Dict
            Mapping of module names to (global input name -> local input name)
        outputs : Dict
            Mapping of module names to (local input name -> global input name)
        """

        log = logger.get_logger()
        self.modules: Dict[str, Tuple[Any, str]] = dict()
        self.inputs = inputs
        self.outputs = outputs

        for name, (type, cfg, action) in modules.items():
            try:
                if issubclass(type, Dataset):
                    self.modules[name] = (type.from_json(cfg), action)

                elif (issubclass(type, ModelWrapper)
                      or issubclass(type, Optimizer)):
                    ds_name = self._find_input_module(name)
                    self.modules[name] = (type.from_json(
                        self.modules[ds_name][0], cfg), action)
            except Exception as e:
                log.error(f'Error loading submodule {name} : {str(e)}')

        self.compile()

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
        return [ds for ds in self.modules if
                set(self.inputs[name]) &
                set(self.outputs[ds].values()) != set()][0]

    def compile(self):
        for name, (module, action) in self.modules.items():
            if issubclass(type(module), Optimizer):
                parent = self._find_input_module(name)
                format = module.consult_model_type(self.modules[parent][0])
                print(format)

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
                        "properties": {
                            "oneOf": [schema for schema in
                                      [Dataset.form_parameterschema(),
                                       ModelWrapper.form_parameterschema(),
                                       Optimizer.form_parameterschema()]]
                        },
                        "additionalProperties": False
                    },
                    'required': ['type', 'parameters']
                }
            }
        }

    @classmethod
    def from_json(cls, json_dict: Dict[str, Any]):
        log = logger.get_logger()

        try:
            jsonschema.validate(json_dict, cls.form_parameterschema())
        except jsonschema.ValidationError:
            log.error('JSON description is invalid')

        modules: Dict[str, Tuple[Type, Any, str]] = dict()
        inputs: Dict[str, Dict[str, str]] = dict()
        outputs: Dict[str, Dict[str, str]] = dict()

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

    def process(self):
        """
        Main process function. Repeatedly fires constructed graph in a loop.
        """
        log = logger.get_logger()
        current_outputs: Dict[str, Any] = dict()
        while True:
            try:
                for name, (module, action) in self.modules.items():
                    input: Dict[str, Any] = {
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

            except KeyboardInterrupt:
                log.warn('Processing interrupted due to keyboard interrupt.\
                      Aborting.')
                break

            except StopIteration:
                log.warn(f'Processing interrupted due to empty {name} stream.\
                    Aborting.')
                break

            except RuntimeError:
                log.warn(
                    f'Processing interrupted from {name} module. Aborting.')
                break

        return current_outputs
