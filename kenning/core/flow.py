from typing import Any, Dict, Tuple, Type
from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper

from kenning.utils.class_loader import load_class


class KenningFlow:

    # Following structure of comments is not final
    # and meant for WIP purposes only
    # modules is Map of module_name -> (Type, Type config, action)
    # inputs is Map module_name -> Map(global -> local_input)
    # outputs is Map module_name -> Map(local_output -> global)
    def __init__(self, modules: Dict[str, Tuple[Type, Any, str]],
                 inputs: Dict[str, Dict[str, str]],
                 outputs: Dict[str, Dict[str, str]]):

        self.nodes: Dict[str, Tuple[Any, str]] = dict()
        self.inputs = inputs
        self.outputs = outputs

        for name, (type, cfg, action) in modules.items():
            if issubclass(type, Dataset):
                self.nodes[name] = (type.from_json(cfg), action)

            elif issubclass(type, ModelWrapper):
                ds_name = [ds for ds in self.nodes if
                           set(inputs[name]) &
                           set(outputs[ds].values()) != set()][0]
                self.nodes[name] = (type.from_json(
                    self.nodes[ds_name][0], cfg), action)

    def compile(self):
        pass

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
        # Map global -> value
        current_outputs: Dict[str, Any] = dict()
        while True:
            try:
                for name, (node, action) in self.nodes.items():
                    input: Dict[str, Any] = {
                        local_name: current_outputs[global_name]
                        for global_name, local_name in
                        self.inputs[name].items()
                    } if name in self.inputs else {}

                    output = node.actions[action](input)

                    current_outputs.update({
                        self.outputs[name][local_output]: value
                        for local_output, value in
                        output.items()
                    })

            except (KeyboardInterrupt, StopIteration):
                break

        return current_outputs
