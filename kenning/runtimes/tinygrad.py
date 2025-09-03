# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime implementation for Tinygrad models.
"""

import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Type

import numpy as np

from kenning.core.exceptions import (
    InputNotPreparedError,
    ModelNotLoadedError,
    ModelNotPreparedError,
    NotSupportedError,
)
from kenning.core.runtime import Runtime
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI, extract_tar

ALL_DEVICES = [
    "METAL",
    "AMD",
    "NV",
    "CUDA",
    "QCOM",
    "GPU",
    "CPU",
    "LLVM",
    "DSP",
    "WEBGPU",
]
DEFAULT_DEVICE = "CPU"


class TinygradRuntime(Runtime):
    """
    Runtime subclass that provides an API
    for testing inference on Tinygrad models.
    """

    inputtypes = ["tinygrad"]
    arguments_structure = {
        "model_path": {
            "argparse_name": "--save-model-path",
            "description": "Path where the model will be uploaded",
            "type": ResourceURI,
            "default": "model.tar",
        },
        "skip_jit": {
            "argparse_name": "--skip-jit",
            "description": "Do not execute JIT compilation of the model",
            "type": bool,
            "default": False,
        },
        "jit_prune": {
            "argparse_name": "--jit-prune",
            "description": "Should JIT prune independent kernels",
            "type": bool,
            "default": False,
        },
        "jit_optimize": {
            "argparse_name": "--jit-optimize",
            "description": "Should JIT optimize buffer memory layout",
            "type": bool,
            "default": False,
        },
        "batch_size": {
            "argparse_name": "--batch-size",
            "description": "JIT batch size",
            "type": int,
            "default": 32,
        },
        "target_device_backend": {
            "argparse_name": "--target-device-backend",
            "description": "What backend should be used",
            "default": DEFAULT_DEVICE,
            "enum": ALL_DEVICES,
        },
        "beam_count": {
            "argparse_name": "--beam-count",
            "description": "Number of beams in kernel beam search",
            "type": int,
            "default": 0,
        },
        "debug_level": {
            "argparse_name": "--debug-level",
            "description": "Debug level used by tinygrad",
            "type": int,
            "default": 0,
        },
        "disable_opt": {
            "argparse_name": "--disable-opt",
            "description": "Disable optimizations",
            "type": bool,
            "default": False,
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        skip_jit: bool = False,
        jit_prune: bool = False,
        jit_optimize: bool = False,
        batch_size: int = 32,
        target_device_backend: str = DEFAULT_DEVICE,
        beam_count: int = 0,
        debug_level: int = 0,
        disable_opt: bool = False,
        disable_performance_measurements: bool = False,
    ):
        self.model_path = model_path
        self.input_spec = None
        self.workdir = None
        self.use_tinyjit = not skip_jit
        self.tinyjit_prune = jit_prune
        self.tinyjit_optimize = jit_optimize
        self.target_backend = target_device_backend
        os.environ[target_device_backend] = "1"
        os.environ["JIT_BATCH_SIZE"] = str(batch_size)
        os.environ["BEAM"] = str(beam_count)
        os.environ["DEBUG"] = str(debug_level)
        if disable_opt:
            os.environ["NOOPT"] = "1"

        super().__init__(
            disable_performance_measurements=disable_performance_measurements
        )

    def load_metadata(self):
        """
        Loads metadata from json file from unpacked model file.
        """
        assert (
            self.workdir is not None
        ), "Unpacking model .tar file is needed first"

        json_filename = self.model_path.with_suffix(".json").name
        with open(self.workdir / json_filename, mode="r") as f:
            info = json.load(f)
            self.metadata = info

    def load_model_class(self) -> Type:
        """
        Loads model class from received implementation metadata.

        Returns
        -------
        Type
            model class that was loaded from unpacked model file.
        """
        assert (
            self.workdir is not None
        ), "Upacking model .tar file is needed first"
        assert self.metadata is not None, "Loading metadata is needed first"

        from kenning.modelwrappers.frameworks.tinygrad import TinygradWrapper
        from kenning.optimizers.tinygrad import TinygradMetadata

        impl_filename = self.metadata[TinygradMetadata.MODELCLS_FILENAME]
        impl_file = self.workdir / impl_filename
        impl_clsname = self.metadata[TinygradMetadata.MODELCLS]
        return TinygradWrapper.load_model_class(impl_clsname, impl_file)

    def load_model(self, modelcls: Type) -> Callable:
        """
        Initializes and returns model object using modelcls class.

        Parameters
        ----------
        modelcls: Type
            model class that will be initialized

        Returns
        -------
        Callable
            returns initialized model that is ready for inference.

        Raises
        ------
        NotSupportedError
            Raised when loaded model_type format is not supported.
        """
        assert (
            self.workdir is not None
        ), "Upacking model .tar file is needed first"
        assert self.metadata is not None, "loading metadata is needed first"

        from tinygrad.nn.state import load_state_dict, safe_load

        from kenning.optimizers.tinygrad import TinygradMetadata

        weights_filename = self.metadata[
            TinygradMetadata.MODELWEIGHTS_FILENAME
        ]
        weights_file = self.workdir / weights_filename

        model_type = self.metadata[TinygradMetadata.MODEL_TYPE]
        self.model_type = model_type

        if model_type == "tinygrad":
            model = modelcls()
            load_state_dict(model, safe_load(weights_file))
            return model
        elif model_type == "onnx":
            model = modelcls(weights_file)
            return model
        else:
            raise NotSupportedError("Other model types are not supported")

    def prepare_model(self, input_data: Optional[bytes]) -> bool:
        from tinygrad import TinyJit

        self.workdir = Path(tempfile.mkdtemp())

        KLogger.info("Extracting data")
        if input_data:
            file_object = io.BytesIO(input_data)
            with tarfile.open(fileobj=file_object) as tar:
                tar_name = tar.getnames()[0]

                for member in tar.getmembers():
                    # Skip the top-level directory
                    if member.path == tar_name:
                        continue

                    # Remove the top-level directory from the path
                    new_name = member.path.removeprefix(tar_name).removeprefix(
                        "/"
                    )
                    member.name = new_name
                    tar.extract(member, path=self.workdir)
        else:
            extract_tar(src_path=self.model_path, target_dir=self.workdir)

        self.load_metadata()
        modelcls = self.load_model_class()

        KLogger.info("Loading model")

        self.model = self.load_model(modelcls)

        if (
            self.tinyjit_prune or self.tinyjit_optimize
        ) and not self.use_tinyjit:
            raise ModelNotLoadedError(
                "Trying to use JIT-specific parameters, having disabled JIT"
            )

        if self.use_tinyjit:
            self.model = TinyJit(
                self.model,
                prune=self.tinyjit_prune,
                optimize=self.tinyjit_optimize,
            )
        KLogger.info("Model loading ended successfully")
        return True

    def load_input(self, input_data: List[np.ndarray]) -> bool:
        from tinygrad import Tensor

        if input_data is None or 0 == len(input_data):
            KLogger.error("Received empty input data")
            return False
        KLogger.debug(f"Loading inputs of size {len(input_data)}")

        self.input = {}
        if self.model_type == "onnx":
            for spec, inp in zip(
                self.processed_input_spec
                if self.processed_input_spec
                else self.input_spec,
                input_data,
            ):
                self.input[spec["name"]] = Tensor(
                    inp.astype(np.float32)
                ).realize()
        elif self.model_type == "tinygrad":
            self.input = [
                Tensor(inp.astype(np.float32)).realize() for inp in input_data
            ]

        return True

    def run(self):
        if self.model is None:
            raise ModelNotPreparedError
        if self.input is None:
            raise InputNotPreparedError

        if self.model_type == "onnx":
            self.output = self.model(self.input)
        elif self.model_type == "tinygrad":
            self.output = [self.model(inp).realize() for inp in self.input]
        self.input = None

    def extract_output(self) -> List[np.ndarray]:
        results = []

        if self.model_type == "onnx":
            for name, value in self.output.items():
                results.append(value.numpy())
        elif self.model_type == "tinygrad":
            # expected postprocessing to make it into numpy arrays
            for out in self.output:
                results.append(out.numpy())

        return results
