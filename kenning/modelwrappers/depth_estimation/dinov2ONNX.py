# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains ONNX model wrapper for dinov2 depth estimation.
"""

import shutil
from typing import Any, List, Optional

import numpy as np
import onnx

from kenning.core.dataset import Dataset
from kenning.core.exceptions import NotSupportedError
from kenning.core.model import ModelWrapper
from kenning.utils.resource_manager import PathOrURI


class Dinov2ONNX(ModelWrapper):
    """
    Model wrapper for depth estimation in ONNX.
    """

    pretrained_model_uri = "kenning:///models/depth_estimation/dinov2.onnx"
    arguments_structure = {
        "input_width": {
            "argparse_name": "--input-width",
            "description": "Input width",
            "type": int,
            "default": 244,
        },
        "input_height": {
            "argparse_name": "--input-height",
            "description": "Input height",
            "type": int,
            "default": 244,
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Dataset,
        from_file: bool = True,
        model_name: Optional[str] = None,
        input_width: int = 244,
        input_height: int = 244,
    ):
        self.original_model_path = model_path
        self.input_width = input_width
        self.input_height = input_height
        self.model_prepared = False
        self.model = None
        super().__init__(model_path, dataset, from_file, model_name)

    @classmethod
    def _get_io_specification(cls, width: int, height: int):
        return {
            "input": [
                {
                    "name": "input",
                    "shape": [1, 3, width, height],
                    "dtype": "float32",
                }
            ],
            "output": [
                {
                    "name": "output",
                    "shape": [1, -1, -1],
                    "dtype": "float32",
                }
            ],
        }

    @classmethod
    def derive_io_spec_from_json_params(cls, json_dict):
        return cls._get_io_specification(-1, -1)

    def get_io_specification_from_model(self):
        return self._get_io_specification(self.input_width, self.input_height)

    def prepare_model(self):
        if self.model_prepared:
            return None
        if not self.from_file:
            raise NotSupportedError(
                "Dinov2 ModelWrapper only supports loading model from a file."
            )
        self.load_model(self.original_model_path)
        self.model_prepared = True

    def get_framework_and_version(self):
        return ("onnx", onnx.__version__)

    @classmethod
    def get_output_formats(cls):
        return ["onnx"]

    def convert_input_to_bytes(self, inputdata: List[np.ndarray]) -> bytes:
        return inputdata[0].tobytes()

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

    def run_inference(self, X: List[Any]) -> List[Any]:
        raise NotSupportedError(
            "To run this model you need to define "
            "ONNXRuntime in the scenario configuration"
        )

    def load_model(self, model_path: PathOrURI):
        if self.model is not None:
            del self.model
        self.model = onnx.load_model(str(model_path))

    def save_to_onnx(self, model_path: PathOrURI):
        self.save_model(model_path)

    def save_model(self, model_path: PathOrURI):
        shutil.copy(self.original_model_path, model_path)

    def train_model(
        self,
        train_loader,
        test_loader,
        opt,
        criterion,
        postprocess,
        metric_func,
        epoch_start_hook,
    ):
        raise NotSupportedError
