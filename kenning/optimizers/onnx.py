# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for ONNX deep learning compiler.
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional

import onnx

from kenning.converters import converter_registry
from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import (
    Optimizer,
)
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class ONNXCompiler(Optimizer):
    """
    The ONNX compiler.
    """

    inputtypes = [
        "keras",
        "torch",
        "tflite",
        "onnx",
        "any",
        "sklearn",
    ]

    outputtypes = ["onnx"]

    arguments_structure = {
        "model_framework": {
            "argparse_name": "--model-framework",
            "description": "The input type of the model, framework-wise",
            "default": "any",
            "enum": inputtypes + ["any"],
        }
    }

    def __init__(
        self,
        dataset: Dataset,
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        compilation_metadata: Optional[Path] = None,
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
        compilation_metadata : Optional[Path]
            Path where compilation metadata will be saved (in JSON format)
            if available.
        model_framework : str
            Framework of the input model, used to select a proper backend. If
            set to "any", then the optimizer will try to derive model framework
            from file extension.
        model_wrapper : Optional[ModelWrapper]
            ModelWrapper for the optimized model (optional).
        """
        self.model_framework = model_framework
        self.set_input_type(model_framework)
        super().__init__(
            dataset,
            compiled_model_path,
            location,
            compilation_metadata,
            model_wrapper,
        )

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
        **kwargs: Dict,
    ):
        input_model_path = ResourceURI(input_model_path)

        if io_spec is None:
            io_spec = self.load_io_specification(input_model_path)

        input_type = self.get_input_type(input_model_path)

        model_cls = self.get_model_class()

        if model_cls is None:
            KLogger.warning("Cannot get model class from model wrapper.")

        conversion_kwargs = {
            "io_spec": io_spec,
            "model_cls": model_cls,
        }

        model = converter_registry.convert(
            input_model_path, input_type, "onnx", **conversion_kwargs, **kwargs
        )

        onnx.save(model, self.compiled_model_path)

        # update the io specification with names
        for spec, input in zip(io_spec["input"], model.graph.input):
            spec["name"] = input.name

        for spec, output in zip(io_spec["output"], model.graph.output):
            spec["name"] = output.name

        self.save_io_specification(input_model_path, io_spec)

    @classmethod
    def get_framework(cls) -> str:
        return "onnx"

    @classmethod
    def get_framework_version(cls) -> str:
        return onnx.__version__
