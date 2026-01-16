# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for IREE compiler.
"""

import re
import subprocess
from typing import Dict, List, Literal, Optional, Tuple

import onnx
from iree.compiler import tools as ireecmp
from iree.compiler import version

from kenning.converters import converter_registry
from kenning.core.dataset import Dataset
from kenning.core.exceptions import (
    CompilationError,
)
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import (
    Optimizer,
)
from kenning.core.platform import Platform
from kenning.utils.logger import KLogger
from kenning.utils.onnx import check_io_spec
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
        "keras": ...,
        "tflite": ...,
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
            "required": False,
            "enum": list(backend_convert.keys()),
        },
        "compiler_args": {
            "argparse_name": "--compiler-args",
            "description": "Additional options that are passed to compiler",
            "type": list[str],
            "default": None,
            "nullable": True,
        },
    }

    def __init__(
        self,
        dataset: Dataset,
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        backend: str = "vmvx",
        model_framework: str = "any",
        compiler_args: Optional[List[str]] = None,
        model_wrapper: Optional[ModelWrapper] = None,
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
        model_wrapper : Optional[ModelWrapper]
            ModelWrapper for the optimized model (optional).
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

        super().__init__(dataset, compiled_model_path, location, model_wrapper)

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

        io_spec_processed = check_io_spec(io_spec)

        conversion_kwargs = {"io_spec": io_spec_processed}

        # To compile a model with IREE compiler, we first convert it to ONNX
        # (that's because IREE TensorFlow workflow, as of version 3.6.0 is
        # highly unstable, so trying to compile directly does not work).
        onnx_model = converter_registry.convert(
            input_model_path, input_type, "onnx", **conversion_kwargs
        )

        intermediate_onnx_model_path = self.compiled_model_path.with_suffix(
            ".tmp.onnx"
        )

        onnx.save(onnx_model, intermediate_onnx_model_path)
        KLogger.debug(
            "Saved model in intermediate onnx format at:"
            f" {intermediate_onnx_model_path}"
        )

        # Compiled IREE models have an entry function, that has to be called to
        # start inference. For compiled onnx models, name of that entry
        # function is the same as the onnx graph name. 'module' is the default
        # IREE bytecode module name.
        io_spec["entry_func"] = "module." + onnx_model.graph.name

        intermediate_mlir_path = self.compiled_model_path.with_suffix(
            ".tmp.mlir"
        )

        subprocess.call(
            [
                "iree-import-onnx",
                intermediate_onnx_model_path.resolve(),
                "--opset-version",
                "17",
                "-o",
                intermediate_mlir_path.resolve(),
            ]
        )

        try:
            compiled_buffer = ireecmp.compile_file(
                str(intermediate_mlir_path.resolve()),
                input_type="onnx",
                extra_args=self.parsed_compiler_args,
                target_backends=[self.converted_backend],
            )
        except ireecmp.CompilerToolError as e:
            raise CompilationError(e)

        with open(self.compiled_model_path, "wb") as f:
            f.write(compiled_buffer)
        self.save_io_specification(self.compiled_model_path, io_spec)

    def get_framework_and_version(self):
        return "iree", version.VERSION

    def read_platform(self, platform: Platform):
        super().read_platform(platform)
        match type(platform).__name__:
            case "CUDAPlatform":
                self.backend = "cuda"
                self.converted_backend = "cuda"
                if platform.compute_capability in [
                    "ada",
                    "hopper",
                    "rtx4090",
                ] or (
                    platform.compute_capability.startswith("sm_")
                    and int(platform.compute_capability.removeprefix("sm_"))
                    > 86
                ):
                    KLogger.warning(
                        f"Platform '{platform.compute_capability}'"
                        " is not supported by this compiler - check"
                        " https://github.com/iree-org/iree/issues/21122."
                        " Use 'sm_86' instead."
                    )
                self.parsed_compiler_args.extend(
                    [
                        "--iree-hal-target-device=cuda",
                        f"--iree-cuda-target={platform.compute_capability}",
                    ]
                )
            case "BareMetalPlatform":
                KLogger.info("BareMetalPlatform support still in development.")
            case _:
                KLogger.warning(
                    f"Unsupported platform: {type(platform).__name__}."
                )
