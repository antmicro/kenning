# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides base methods for using tinygrad models in Kenning.
"""

import importlib.metadata
import importlib.util
import sys
from abc import ABC
from typing import Any, Callable, List, Optional, Tuple, Type

import numpy as np

from kenning.core.dataset import Dataset
from kenning.core.exceptions import ModelNotLoadedError, NotSupportedError
from kenning.core.model import ModelWrapper
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class TinygradWrapper(ModelWrapper, ABC):
    """
    Base model wrapper for tinygrad models.
    """

    arguments_structure = {
        "model_file": {
            "argparse_name": "--model-file",
            "description": "Path of file containing python implementation of model",  # noqa: E501
            "type": str,
            "default": None,
        },
        "modelcls": {
            "argparse_name": "--model-cls",
            "description": "Entry point for implementation (should be class inside implementation file)",  # noqa: E501
            "type": str,
            "default": "MainModule",
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Dataset,
        model_file: str = None,
        modelcls: str = "MainModule",
        from_file: bool = True,
        model_name: Optional[str] = None,
    ):
        super().__init__(model_path, dataset, from_file, model_name)
        self.model = None
        self.model_file = model_file
        self.modelcls_name = modelcls

    @staticmethod
    def load_weights(modelcls: Type, weights_path: PathOrURI) -> Callable:
        """
        Load safetensors to model that will be initialized from modelcls class.

        Parameters
        ----------
        modelcls: Type
            Class that will be initialized and loaded.
        weights_path: PathOrURI
            Weights file that will be used to initialize model.

        Returns
        -------
        Callable
            Callable, initialized model.
        """
        from tinygrad.nn.state import load_state_dict, safe_load

        model = modelcls()
        load_state_dict(model, safe_load(weights_path))
        return model

    @staticmethod
    def load_model_class(modelcls_name: str, module_file: PathOrURI) -> Type:
        """
        Loads class from file.

        Parameters
        ----------
        modelcls_name: str
            Name of the class that is to be loaded.
        module_file: PathOrURI
            File containing class specification.

        Returns
        -------
        Type
            Class loaded from module_file.

        Raises
        ------
        ModelNotLoadedError
            Raised when class could not be loaded from file.
        """
        spec = importlib.util.spec_from_file_location(
            modelcls_name, str(module_file)
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[modelcls_name] = module
        spec.loader.exec_module(module)
        cls_type = getattr(module, modelcls_name, None)
        if cls_type is None:
            raise ModelNotLoadedError(
                "Error occurred when importing tinygrad implementation"
            )
        return cls_type

    def get_model_structure_info(self) -> Tuple[PathOrURI, str]:
        """
        Get info needed to load this model's structure.

        Returns
        -------
        PathOrURI
            File containing model class implementation.
        str
            Name of the model class inside implementation file.
        """
        return self.model_file, self.modelcls_name

    def load_model(self, model_path: PathOrURI):
        from tinygrad import TinyJit

        model_file = ResourceURI(self.model_file)

        modelcls = self.load_model_class(self.modelcls_name, model_file)
        self.model = self.load_weights(modelcls, self.model_path)
        self.jmodel = TinyJit(self.model, optimize=True, prune=True)

    def save_to_onnx(self, model_path: PathOrURI):
        raise NotSupportedError

    def save_model(self, model_path: PathOrURI):
        from tinygrad.nn.state import get_state_dict, safe_save

        safe_save(get_state_dict(self.model), model_path)

    def run_inference(self, X: List[np.ndarray]) -> List[Any]:
        from tinygrad import Tensor

        self.prepare_model()
        input = [Tensor(inp.astype(np.float32)).realize() for inp in X]
        y = [self.jmodel(inp).realize().numpy() for inp in input]
        return y

    def get_framework_and_version(self) -> Tuple[str, str]:
        return ("tinygrad", importlib.metadata.version("tinygrad"))

    @classmethod
    def get_output_formats(cls) -> List[str]:
        return ["safetensors"]

    def convert_input_to_bytes(self, inputdata: List[Any]) -> bytes:
        data = bytes()
        for inp in inputdata:
            for x in inp:
                data += x.tobytes()
        return data

    def convert_output_from_bytes(self, outputdata: bytes) -> List[Any]:
        out_spec = self.get_io_specification()["output"]

        result = []
        data_idx = 0
        for spec in out_spec:
            dtype = np.dtype(spec["dtype"])
            shape = spec["shape"]

            out_size = np.prod(shape) * np.dtype(dtype).itemsize
            arr = np.frombuffer(
                outputdata[data_idx : data_idx + out_size], dtype=dtype
            )
            data_idx += out_size
            result.append(arr.reshape(shape))

        return result

    def train_model(self):
        raise NotSupportedError
