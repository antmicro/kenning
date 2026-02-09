# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for TVM deep learning compiler.
"""

import importlib.metadata
import json
import shutil
import tempfile

try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        """
        StrEnum stand-in for python 3.10.
        """

        pass


from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from kenning.converters import converter_registry
from kenning.core.dataset import Dataset
from kenning.core.exceptions import IOSpecificationNotFoundError
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import (
    PathOrURI,
    ResourceManager,
    create_tar,
)


class TinygradMetadata(StrEnum):
    """
    Enum class containing all metadata keys for tinygrad json info file.
    """

    MODELCLS = "modelcls_name"
    MODELCLS_FILENAME = "model_impl_filename"
    MODELWEIGHTS_FILENAME = "model_weights_filename"
    MODEL_TYPE = "model_type"


class TinygradOptimizer(Optimizer):
    """
    Runtime subclass that provides an API
    for testing inference on Tinygrad models.
    """

    inputtypes = [
        "onnx",
        "safetensors",
        "any",
    ]

    outputtypes = ["tinygrad"]

    arguments_structure = {
        "model_framework": {
            "argparse_name": "--model-framework",
            "description": "The input type of the model, framework-wise",
            "default": "any",
            "enum": inputtypes + ["any"],
        },
    }

    def __init__(
        self,
        dataset: Dataset,
        compiled_model_path: PathOrURI,
        model_framework: str = "any",
        location: Literal["host", "target"] = "host",
        model_wrapper: Optional[ModelWrapper] = None,
    ):
        self.model_path = compiled_model_path
        self.input_spec = None
        self.model_wrapper = model_wrapper
        self.model_framework = model_framework
        self.set_input_type(model_framework)

        super().__init__(dataset, compiled_model_path, location, model_wrapper)

    def get_complete_model_structure_info(self) -> Tuple[str, str, str]:
        """
        Get info needed to load this model's structure.

        Returns
        -------
        str
            File containing model class implementation.
        str
            Name of the model class inside implementation file.
        str
            Type of framework that will be used
        """
        import tinygrad.frontend.onnx

        from kenning.modelwrappers.frameworks.tinygrad import TinygradWrapper

        if issubclass(type(self.model_wrapper), TinygradWrapper):
            (
                model_module_path,
                modelcls_name,
            ) = self.model_wrapper.get_model_structure_info()
            return (model_module_path, modelcls_name, "tinygrad")
        else:
            return (tinygrad.frontend.onnx.__file__, "OnnxRunner", "onnx")

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
        **kwargs: Dict,
    ):
        import onnx

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

        model_cls = self.get_model_class()

        if model_cls is None:
            KLogger.warning("Cannot get model class from model wrapper.")

        conversion_kwargs = {
            "io_spec": io_spec,
            "model_cls": model_cls,
        }

        model = converter_registry.convert(
            input_model_path,
            input_type,
            "tinygrad",
            **conversion_kwargs,
            **kwargs,
        )
        onnx_path = self.compiled_model_path.with_suffix(".onnx")
        onnx.save(model, onnx_path)

        input_model_path = onnx_path

        for spec, input in zip(io_spec["input"], model.graph.input):
            spec["name"] = input.name

        for spec, output in zip(io_spec["output"], model.graph.output):
            spec["name"] = output.name

        self.save_io_specification(input_model_path, io_spec)
        (
            model_module_path,
            modelcls_name,
            model_type,
        ) = self.get_complete_model_structure_info()
        model_module_path = ResourceManager().get_resource(
            uri=model_module_path
        )

        model_weights_filename = "model_weights"
        model_implementation_filename = "implementation.py"
        metadata = {
            TinygradMetadata.MODELCLS: modelcls_name,
            TinygradMetadata.MODELCLS_FILENAME: model_implementation_filename,
            TinygradMetadata.MODELWEIGHTS_FILENAME: model_weights_filename,
            TinygradMetadata.MODEL_TYPE: model_type,
        }
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir_path = Path(tmpdirname)
            metadata_filename = (
                tmpdir_path / self.model_path.with_suffix(".json").name
            )

            with open(metadata_filename, "w") as f:
                json.dump(metadata, f)
            shutil.copy(input_model_path, tmpdir_path / model_weights_filename)
            shutil.copy(
                model_module_path,
                tmpdir_path / model_implementation_filename,
            )
            create_tar(self.model_path, tmpdirname)
        self.save_io_specification(input_model_path, io_spec)

    @classmethod
    def get_framework(cls) -> str:
        return "tinygrad"

    @classmethod
    def get_framework_version(cls) -> str:
        return importlib.metadata.version("tinygrad")
