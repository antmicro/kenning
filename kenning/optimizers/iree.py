# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for IREE compiler.
"""

import re
from typing import Dict, List, Literal, Optional, Tuple

from iree.compiler import tools as ireecmp
from iree.compiler import version

from kenning.core.dataset import Dataset
from kenning.core.optimizer import (
    CompilationError,
    IOSpecificationNotFoundError,
    Optimizer,
)
from kenning.utils.resource_manager import PathOrURI


def input_shapes_dict_to_list(
    inputshapes: Dict[str, Tuple[int, ...]],
) -> List[Tuple[int, ...]]:
    """
    Turn the dictionary of 'name':'shape' of every input layer to ordered list.
    The order of input layers is inferred from names. It is assumed that the
    name of every input layer contains single ID number, and the order of the
    inputs are according to their IDs.

    Parameters
    ----------
    inputshapes : Dict[str, Tuple[int, ...]]
        The inputshapes argument of IREECompiler.compile method.

    Returns
    -------
    List[Tuple[int, ...]]
        Shapes of each input layer in order.
    """
    layer_order = {}
    for name in inputshapes.keys():
        layer_id = int(re.search(r"\d+", name).group(0))
        layer_order[name] = layer_id
    ordered_layers = sorted(list(inputshapes.keys()), key=layer_order.get)
    return [inputshapes[layer] for layer in ordered_layers]


def kerasconversion(model_path: PathOrURI, input_spec: Dict) -> bytes:
    """
    Converts the Keras model to IREE.

    Parameters
    ----------
    model_path: PathOrURI
        Path to the model to convert
    input_spec: Dict
        Provides the specification of the inputs

    Returns
    -------
    bytes
        Compiled model
    """
    import tensorflow as tf
    from iree.compiler import tf as ireetf

    # Calling the .fit() method of keras model taints the state of the model,
    # breaking the IREE compiler. Because of that, the workaround is needed.
    original_model = tf.keras.models.load_model(str(model_path), compile=False)
    model = tf.keras.models.clone_model(original_model)
    model.set_weights(original_model.get_weights())
    del original_model

    inputspec = [
        tf.TensorSpec(spec["shape"], spec["dtype"]) for spec in input_spec
    ]

    class WrapperModule(tf.Module):
        def __init__(self):
            super().__init__()
            self.m = model
            self.m.main = lambda *args: self.m(*args, training=False)
            self.main = tf.function(input_signature=inputspec)(self.m.main)

    return ireetf.compile_module(
        WrapperModule(), exported_names=["main"], import_only=True
    )


def tensorflowconversion(model_path: PathOrURI, input_spec: Dict) -> bytes:
    """
    Converts the TensorFlow model to IREE.

    Parameters
    ----------
    model_path: PathOrURI
        Path to the model to convert
    input_spec: Dict
        Provides the specification of the inputs

    Returns
    -------
    bytes
        A bytes-like object with the compiled model
    """
    import tensorflow as tf
    from iree.compiler import tf as ireetf

    model = tf.saved_model.load(model_path)

    inputspec = [
        tf.TensorSpec(spec["shape"], spec["dtype"]) for spec in input_spec
    ]

    model.main = tf.function(input_signature=inputspec)(
        lambda *args: model(*args)
    )
    return ireetf.compile_module(
        model, exported_names=["main"], import_only=True
    )


def tfliteconversion(model_path: PathOrURI, input_spec: Dict) -> bytes:
    """
    Converts the TFLite model to IREE.

    Parameters
    ----------
    model_path: PathOrURI
        Path to the model to convert
    input_spec: Dict
        Provides the specification of the inputs

    Returns
    -------
    bytes
        A bytes-like object with the compiled output
    """
    from iree.compiler import tflite as ireetflite

    return ireetflite.compile_file(str(model_path), import_only=True)


backend_convert = {
    # CPU backends
    "dylib": "dylib-llvm-aot",
    "vmvx": "vmvx",
    # GPU backends
    "vulkan": "vulkan-spirv",
    "cuda": "cuda",
    "llvm-cpu": "llvm-cpu",
}


class IREECompiler(Optimizer):
    """
    IREE compiler.
    """

    inputtypes = {
        "keras": kerasconversion,
        "tensorflow": tensorflowconversion,
        "tflite": tfliteconversion,
    }

    outputtypes = ["iree"]

    arguments_structure = {
        "model_framework": {
            "argparse_name": "--model-framework",
            "description": "The input type of the model, framework-wise",
            "default": "any",
            "enum": list(inputtypes.keys()) + ["any"],
        },
        "backend": {
            "argparse_name": "--backend",
            "description": "Name of the backend that will run the compiled module",  # noqa: E501
            "required": True,
            "enum": list(backend_convert.keys()),
        },
        "compiler_args": {
            "argparse_name": "--compiler-args",
            "description": "Additional options that are passed to compiler",
            "default": None,
            "is_list": True,
            "nullable": True,
        },
    }

    def __init__(
        self,
        dataset: Dataset,
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        backend: str = "vmvx",
        model_framework: str = "keras",
        compiler_args: Optional[List[str]] = None,
    ):
        """
        Wrapper for IREE compiler.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the model - may be used for quantization
            during compilation stage.
        compiled_model_path : PathOrURI
            Path or URI where compiled model will be saved.
        location : Literal['host', 'target']
            Specifies where optimization should be performed in client-server
            scenario.
        backend : str
            Backend on which the model will be executed.
        model_framework : str
            Framework of the input model, used to select a proper backend. If
            set to "any", then the optimizer will try to derive model framework
            from file extension.
        compiler_args : Optional[List[str]]
            Additional arguments for the compiler. Every options should be in a
            separate string, which should be formatted like this:
            <option>=<value>, or <option> for flags (example:
            'iree-cuda-llvm-target-arch=sm_60'). Full list of options can be
            listed by running 'iree-compile -h'.
        """
        self.model_framework = model_framework
        self.set_input_type(model_framework)
        self.backend = backend
        self.compiler_args = compiler_args

        self.converted_backend = backend_convert.get(backend, backend)
        if compiler_args is not None:
            self.parsed_compiler_args = [
                f"--{option}" for option in compiler_args
            ]
        else:
            self.parsed_compiler_args = []

        self.set_input_type(model_framework)

        super().__init__(
            dataset=dataset,
            compiled_model_path=compiled_model_path,
            location=location,
        )

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ):
        if io_spec is None:
            io_spec = self.load_io_specification(input_model_path)

        input_type = self.get_input_type(input_model_path)

        if input_type in ("keras", "tensorflow"):
            self.compiler_input_type = "mhlo"
        elif input_type == "tflite":
            self.compiler_input_type = "tosa"

        try:
            input_spec = (
                io_spec["processed_input"]
                if "processed_input" in io_spec
                else io_spec["input"]
            )
        except (TypeError, KeyError):
            raise IOSpecificationNotFoundError("No input specification found")

        imported_model = self.inputtypes[input_type](
            input_model_path, input_spec
        )
        try:
            compiled_buffer = ireecmp.compile_str(
                imported_model,
                input_type=self.compiler_input_type,
                extra_args=self.parsed_compiler_args,
                target_backends=[self.converted_backend],
            )
        except ireecmp.CompilerToolError as e:
            raise CompilationError(e)

        with open(self.compiled_model_path, "wb") as f:
            f.write(compiled_buffer)
        self.save_io_specification(input_model_path, io_spec)

    def get_framework_and_version(self):
        return "iree", version.VERSION
