# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for ONNX deep learning compiler.
"""

from typing import Dict, List, Literal, Optional

import onnx

from kenning.converters import converter_registry
from kenning.core.dataset import Dataset
from kenning.core.exceptions import (
    IOSpecificationNotFoundError,
)
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import (
    Optimizer,
)
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class ONNXCompiler(Optimizer):
    """
    The ONNX compiler.
    """

    inputtypes = {
        "keras": ...,
        "torch": ...,
        "tflite": ...,
        "onnx": ...,
    }

    outputtypes = ["onnx"]

    arguments_structure = {
        "model_framework": {
            "argparse_name": "--model-framework",
            "description": "The input type of the model, framework-wise",
            "default": "any",
            "enum": list(inputtypes.keys()) + ["any"],
        }
    }

    def __init__(
        self,
        dataset: Dataset,
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        model_framework: str = "any",
        model_wrapper: Optional[ModelWrapper] = None,
    ):
        """
        The ONNX compiler.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the model.
        compiled_model_path : PathOrURI
            Path or URI where compiled model will be saved.
        location : Literal['host', 'target']
            Specifies where optimization should be performed in client-server
            scenario.
        model_framework : str
            Framework of the input model, used to select a proper backend. If
            set to "any", then the optimizer will try to derive model framework
            from file extension.
        model_wrapper : Optional[ModelWrapper]
            ModelWrapper for the optimized model (optional).
        """
        self.model_framework = model_framework
        self.set_input_type(model_framework)
        super().__init__(dataset, compiled_model_path, location, model_wrapper)

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ):
        input_model_path = ResourceURI(input_model_path)

        if io_spec is None:
            io_spec = self.load_io_specification(input_model_path)

        try:
            from copy import deepcopy

            io_spec = deepcopy(io_spec)

            io_spec["input"] = (
                io_spec["processed_input"]
                if "processed_input" in io_spec
                else io_spec["input"]
            )

        except (TypeError, KeyError):
            raise IOSpecificationNotFoundError(
                "No input/output specification found"
            )

        input_type = self.get_input_type(input_model_path)
        conversion_kwargs = {
            "io_spec": io_spec,
        }
        model = converter_registry.convert(
            input_model_path, input_type, "onnx", **conversion_kwargs
        )

        onnx.save(model, self.compiled_model_path)

        # update the io specification with names
        for spec, input in zip(io_spec["input"], model.graph.input):
            spec["name"] = input.name

        for spec, output in zip(io_spec["output"], model.graph.output):
            spec["name"] = output.name

        self.save_io_specification(input_model_path, io_spec)

    def get_framework_and_version(self):
        return ("onnx", onnx.__version__)
