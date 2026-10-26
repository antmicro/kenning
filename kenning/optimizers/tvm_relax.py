# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for TVM Relax IR compiler.
"""

from typing import Dict, List, Literal, Optional, Tuple

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI


class TVMRelaxCompiler(Optimizer):
    """
    The TVM Relax IR compiler.
    """

    inputtypes = [
        "onnx",
        "tflite",
        "any",
    ]

    outputtypes = ["tvm_relax"]

    arguments_structure = {
        "model_framework": {
            "description": "The input type of the model, framework-wise",
            "default": "any",
            "enum": inputtypes,
        },
        "target": {
            "description": "The kind of the target device",
            "default": "llvm",
        },
    }

    def __init__(
        self,
        dataset: Dataset,
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        model_framework: str = "any",
        target: str = "llvm",
        model_wrapper: Optional[ModelWrapper] = None,
    ):
        """
        A TVM Relax compiler wrapper.

        Parameters
        ----------
        dataset : Dataset
            Dataset object.
        compiled_model_path : PathOrURI
            Path where compiled model will be saved.
        location : Literal['host', 'target']
            Specifies where optimization should be performed in client-server
            scenario.
        model_framework : str
            Framework of the input model, used to select a proper backend. If
            set to "any", then the optimizer will try to derive model framework
            from file extension.
        target : str
            The kind of the target device.
        model_wrapper : Optional[ModelWrapper]
            ModelWrapper for the optimized model (optional).
        """
        self.model_framework = model_framework
        self.set_input_type(model_framework)
        self.target = target
        self.target_obj = None
        super().__init__(dataset, compiled_model_path, location, model_wrapper)

    def init(self):
        import tvm

        self.target_obj = tvm.target.Target({"kind": self.target})
        KLogger.debug(f"Using target: {self.target_obj}")

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
        **kwargs: Dict,
    ):
        raise NotImplementedError

    def get_framework_and_version(self) -> Tuple[str, str]:
        return (self.get_framework(), self.get_framework_version())

    def get_framework(self) -> str:
        return "tvm_relax"

    @classmethod
    def get_framework_version(cls) -> str:
        import tvm

        return tvm.__version__
