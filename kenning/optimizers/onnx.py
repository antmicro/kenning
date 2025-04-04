# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for ONNX deep learning compiler.
"""

from typing import Any, Dict, List, Literal, Optional

import onnx

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import (
    CompilationError,
    ConversionError,
    IOSpecificationNotFoundError,
    Optimizer,
)
from kenning.utils.resource_manager import PathOrURI, ResourceURI


def kerasconversion(
    model_path: PathOrURI, input_spec: Dict, output_names: List
) -> Any:
    """
    Converts Keras model to ONNX.

    Parameters
    ----------
    model_path: PathOrURI
        Path to the model to convert
    input_spec: Dict
        Dictionary representing inputs
    output_names: List
        Names of outputs to include in the final model

    Returns
    -------
    Any
        Loaded ONNX model, a variant of ModelProto
    """
    import tensorflow as tf
    import tf2onnx

    model = tf.keras.models.load_model(str(model_path), compile=False)

    input_spec = [
        tf.TensorSpec(spec["shape"], spec["dtype"], name=spec["name"])
        for spec in input_spec
    ]
    modelproto, _ = tf2onnx.convert.from_keras(
        model, input_signature=input_spec
    )

    return modelproto


def torchconversion(
    model_path: PathOrURI, input_spec: Dict, output_names: List
) -> Any:
    """
    Converts Torch model to ONNX.

    Parameters
    ----------
    model_path: PathOrURI
        Path to the model to convert
    input_spec: Dict
        Dictionary representing inputs
    output_names: List
        Names of outputs to include in the final model

    Returns
    -------
    Any
        Loaded ONNX model, a variant of ModelProto

    Raises
    ------
    CompilationError
        Raised if the input type of the model is not torch.nn.Module
    """
    import torch

    dev = "cpu"
    model = torch.load(str(model_path), map_location=dev)

    if not isinstance(model, torch.nn.Module):
        raise CompilationError(
            f"ONNX compiler expects the input data of type: torch.nn.Module, but got: {type(model).__name__}"  # noqa: E501
        )

    model.eval()

    input = tuple(
        torch.randn(spec["shape"], device=dev) for spec in input_spec
    )

    import io

    mem_buffer = io.BytesIO()
    torch.onnx.export(
        model,
        input,
        mem_buffer,
        opset_version=11,
        input_names=[spec["name"] for spec in input_spec],
        output_names=output_names,
    )
    onnx_model = onnx.load_model_from_string(mem_buffer.getvalue())
    return onnx_model


def tfliteconversion(
    model_path: PathOrURI, input_spec: Dict, output_names: List
) -> Any:
    """
    Converts TFLite model to ONNX.

    Parameters
    ----------
    model_path: PathOrURI
        Path to the model to convert
    input_spec: Dict
        Dictionary representing inputs
    output_names: List
        Names of outputs to include in the final model

    Returns
    -------
    Any
        Loaded ONNX model, a variant of ModelProto

    Raises
    ------
    ConversionError
        Raised when model could not be loaded
    """
    import tf2onnx

    try:
        modelproto, _ = tf2onnx.convert.from_tflite(str(model_path))
    except ValueError as e:
        raise ConversionError(e)

    return modelproto


class ONNXCompiler(Optimizer):
    """
    The ONNX compiler.
    """

    inputtypes = {
        "keras": kerasconversion,
        "torch": torchconversion,
        "tflite": tfliteconversion,
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
            The model wrapper object that is optionally used for optimization.
        """
        self.model_framework = model_framework
        self.set_input_type(model_framework)
        super().__init__(dataset, compiled_model_path, location, model_wrapper)
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

            input_spec = (
                io_spec["processed_input"]
                if "processed_input" in io_spec
                else io_spec["input"]
            )
            output_spec = io_spec["output"]
        except (TypeError, KeyError):
            raise IOSpecificationNotFoundError(
                "No input/output specification found"
            )

        try:
            output_names = [spec["name"] for spec in output_spec]
        except KeyError:
            output_names = None

        input_type = self.get_input_type(input_model_path)

        model = self.inputtypes[input_type](
            input_model_path, input_spec, output_names
        )

        onnx.save(model, self.compiled_model_path)

        # update the io specification with names
        for spec, input in zip(input_spec, model.graph.input):
            spec["name"] = input.name

        for spec, output in zip(output_spec, model.graph.output):
            spec["name"] = output.name

        self.save_io_specification(
            input_model_path, {"input": input_spec, "output": output_spec}
        )
        return 0

    def get_framework_and_version(self):
        return ("onnx", onnx.__version__)
