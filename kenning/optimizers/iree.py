# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for IREE compiler.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Literal, Optional, Tuple

import onnx

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
    "llvm-cpu": "llvm-cpu",
    # GPU backends
    "vulkan": "vulkan-spirv",
    "cuda": "cuda",
    # NPU backends
    "coralnpu": "coralnpu",
}


class IREECompiler(Optimizer):
    """
    IREE compiler.
    """

    inputtypes = [
        "flax",
        "keras",
        "tflite",
        "any",
    ]

    outputtypes = ["iree"]

    arguments_structure = {
        "model_framework": {
            "argparse_name": "--model-framework",
            "description": "The input type of the model, framework-wise",
            "default": "any",
            "enum": inputtypes + ["any"],
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
        "compiler_path": {
            "argparse_name": "--compiler-path",
            "description": "Path to the compiler executable",
            "type": list[Path],
            "default": None,
            "nullable": True,
        },
        "linker_path": {
            "argparse_name": "--linker-path",
            "description": "Path to the linker executable",
            "type": list[Path],
            "default": None,
            "nullable": True,
        },
    }

    def __init__(
        self,
        dataset: Dataset,
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        compilation_metadata: Optional[Path] = None,
        backend: Optional[str] = None,
        model_framework: str = "any",
        compiler_args: Optional[List[str]] = None,
        compiler_path: Optional[Path] = None,
        linker_path: Optional[Path] = None,
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
        compilation_metadata : Optional[Path]
            Path where compilation metadata will be saved (in JSON format)
            if available.
        backend : Optional[str]
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
        compiler_path : Optional[Path]
            Path to the compiler executable.
        linker_path : Optional[Path]
            Path to the linker executable.
        model_wrapper : Optional[ModelWrapper]
            ModelWrapper for the optimized model (optional).
        """
        self.model_framework = model_framework
        self.set_input_type(model_framework)
        self.backend = backend
        self.platform_backend = None
        self.compiler_args = compiler_args
        self.compiler_path = compiler_path
        self.linker_path = linker_path

        if compiler_args is not None:
            self.parsed_compiler_args = [
                f"--{option}" for option in compiler_args
            ]
        else:
            self.parsed_compiler_args = []

        if self.compiler_path is None:
            self.compiler_path = os.environ.get("IREE_COMPILER_PATH")

        if self.linker_path is None:
            self.linker_path = os.environ.get("IREE_LINKER_PATH")

        self._tmp_dir = None
        self._tmp_alloc_report = None
        self._tmp_affinity_report = None

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
        if io_spec is None:
            io_spec = self.load_io_specification(input_model_path)

        backend = self._get_backend()

        try:
            input_type = self.get_input_type(input_model_path)
        except Exception:
            input_type = self.model_wrapper.get_framework()

        if input_type in ("keras", "tensorflow"):
            self.compiler_input_type = "mhlo"
        elif input_type == "tflite":
            self.compiler_input_type = "tosa"

        model_cls = self.get_model_class()

        if model_cls is None:
            KLogger.warning("Cannot get model class from model wrapper.")

        intermediate_mlir_path = self.compiled_model_path.with_suffix(
            ".tmp.mlir"
        )

        if input_type == "flax":
            conversion_input = self.model_wrapper or input_model_path
            mlir_model = converter_registry.convert(
                conversion_input,
                "flax",
                "mlir",
            )

            intermediate_mlir_path.write_text(mlir_model)

        else:
            conversion_kwargs = {
                "io_spec": io_spec,
                "model_cls": model_cls,
            }

            # To compile a model with IREE compiler, we first convert it to
            # ONNX (that's because IREE TensorFlow workflow, as of version
            # 3.6.0 is highly unstable, so trying to compile directly does not
            # work).
            onnx_model = converter_registry.convert(
                input_model_path,
                input_type,
                "iree",
                **conversion_kwargs,
                **kwargs,
            )

            intermediate_onnx_model_path = (
                self.compiled_model_path.with_suffix(".tmp.onnx")
            )

            onnx.save(onnx_model, intermediate_onnx_model_path)
            KLogger.debug(
                "Saved model in intermediate onnx format at:"
                f" {intermediate_onnx_model_path}"
            )

            # Compiled IREE models have an entry function, that has to be
            # called to start inference. For compiled onnx models, name of that
            # entry function is the same as the onnx graph name. 'module' is
            # the default IREE bytecode module name.
            io_spec["entry_func"] = "module." + onnx_model.graph.name

            cmd = [
                "iree-import-onnx",
                str(intermediate_onnx_model_path.resolve()),
            ]

            if sys.version_info < (3, 12):
                cmd.extend(["--opset-version", "17"])

            cmd.extend(["-o", str(intermediate_mlir_path.resolve())])

            subprocess.call(cmd)

        if self.compiler_path:
            cmd = [
                self.compiler_path,
                *self.parsed_compiler_args,
                str(intermediate_mlir_path),
                "-o",
                str(self.compiled_model_path),
            ]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError) as e:
                error = (
                    e.stderr
                    if isinstance(e, subprocess.CalledProcessError)
                    else str(e)
                )
                if self._tmp_dir:
                    self._tmp_dir.cleanup()
                raise CompilationError(error or str(e)) from e

        else:
            from iree.compiler import tools as ireecmp

            try:
                compiled_buffer = ireecmp.compile_file(
                    str(intermediate_mlir_path.resolve()),
                    input_type="onnx",
                    extra_args=self.parsed_compiler_args,
                    target_backends=[backend_convert.get(backend, backend)],
                )
            except ireecmp.CompilerToolError as e:
                if self._tmp_dir:
                    self._tmp_dir.cleanup()
                raise CompilationError(e) from e

            with open(self.compiled_model_path, "wb") as f:
                f.write(compiled_buffer)

        # Gather compilation metadata
        if self.compilation_metadata and self._tmp_dir:
            metadata = {"register_allocation": {}}
            if self._tmp_affinity_report.exists():
                with self._tmp_affinity_report.open("r") as fd:
                    metadata |= json.load(fd)
            for root, _, files in self._tmp_alloc_report.walk():
                for f in files:
                    with (root / f).open("r") as fd:
                        metadata["register_allocation"][
                            f.removesuffix(".json")
                        ] = json.load(fd)
            with self.compilation_metadata.open("w") as fd:
                json.dump(metadata, fd)
            self._tmp_dir.cleanup()

        self.save_io_specification(self.compiled_model_path, io_spec)

    @classmethod
    def get_framework(cls):
        return "iree"

    @classmethod
    def get_framework_version(cls) -> str:
        try:
            from iree.compiler import version

            return version.VERSION
        except Exception:
            return "unknown"

    def read_platform(self, platform: Platform):
        super().read_platform(platform)
        match type(platform).__name__:
            case "CUDAPlatform":
                self.platform_backend = "cuda"
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
            case "CoralNPUPlatform":
                if not self.parsed_compiler_args:
                    if flags := getattr(platform, "compilation_flags", None):
                        self.parsed_compiler_args = flags
                        KLogger.debug(
                            "Reading compilation flags from CoralNPU platform "
                            f"{flags}"
                        )
                if self.linker_path:
                    linker_flag = (
                        f"--coralnpu-embedded-linker-path={self.linker_path}"
                    )
                    self.parsed_compiler_args.append(linker_flag)
                    KLogger.debug(f"Added compilation flag {linker_flag}")

                if self.compilation_metadata:
                    self._tmp_dir = TemporaryDirectory(delete=False)
                    self._tmp_affinity_report = (
                        Path(self._tmp_dir.name) / "affinity_profile.json"
                    )
                    self._tmp_alloc_report = (
                        Path(self._tmp_dir.name) / "register_allocation"
                    )
                    self._tmp_alloc_report.mkdir()

                    self.parsed_compiler_args += [
                        "--coralnpu-dump-affinity-profile-format=json",
                        f"--coralnpu-dump-affinity-profile-file={self._tmp_affinity_report}",
                        "--coralnpu-dump-register-allocation-report-format=json",
                        f"--coralnpu-dump-register-allocation-report-dir={self._tmp_alloc_report}",
                    ]
            case _:
                KLogger.warning(
                    f"Unsupported platform: {type(platform).__name__}."
                )

    def _get_backend(self):
        return self.backend or self.platform_backend or "vmvx"
