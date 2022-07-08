from typing import Any, Dict, Tuple, Type
from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer

from kenning.utils.class_loader import load_class


class KenningFlow:

    # Following structure of comments is not final
    # and meant for WIP purposes only
    # modules is Map of module_name -> (module, action)
    # inputs is Map module_name -> Map(global -> local_input)
    # outputs is Map module_name -> Map(local_output -> global)

    def find_input_module(self, name: str) -> str:
        return [ds for ds in self.modules if
                set(self.inputs[name]) &
                set(self.outputs[ds].values()) != set()][0]

    def __init__(self, modules: Dict[str, Tuple[Type, Any, str]],
                 inputs: Dict[str, Dict[str, str]],
                 outputs: Dict[str, Dict[str, str]]):

        self.modules: Dict[str, Tuple[Any, str]] = dict()
        self.inputs = inputs
        self.outputs = outputs

        for name, (type, cfg, action) in modules.items():
            if issubclass(type, Dataset):
                self.modules[name] = (type.from_json(cfg), action)

            elif issubclass(type, ModelWrapper) or issubclass(type, Optimizer):
                ds_name = self.find_input_module(name)
                self.modules[name] = (type.from_json(
                    self.modules[ds_name][0], cfg), action)

        self.compile()

    def compile(self):
        for name, (module, action) in self.modules.items():
            if issubclass(type(module), Optimizer):
                parent = self.find_input_module(name)
                format = module.consult_model_type(self.modules[parent][0])
                print(format)

    @classmethod
    def from_json(cls, args: Dict[str, Any]):
        modules: Dict[str, Tuple[Type, Any, str]] = dict()
        inputs: Dict[str, Dict[str, str]] = dict()
        outputs: Dict[str, Dict[str, str]] = dict()

        for module_name, module_cfg in args.items():
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
                print('Processing interrupted due to keyboard interrupt.\
                      Aborting.')
                break

            except StopIteration:
                print(f'Processing interrupted due to empty {name} stream.\
                    Aborting.')
                break

            except RuntimeError:
                print(f'Processing interrupted from {name} module. Aborting.')
                break

        return current_outputs
