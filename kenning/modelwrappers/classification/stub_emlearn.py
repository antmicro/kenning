# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing an implementation of a stub emlearn model.
"""
from typing import Any

import numpy as np

from kenning.core.model import ModelWrapper


class StubEmlearnModel(ModelWrapper):
    """
    Stub emlearn model mainly used for anomaly detection.
    """

    arguments_structure = {
        "in_features": {
            "argparse_name": "--in_features",
            "type": int,
            "required": True,
        },
        "out_features": {
            "argparse_name": "--out_features",
            "type": int,
            "required": True,
        },
    }

    def __init__(
        self,
        model_path,
        dataset,
        from_file=True,
        model_name=None,
        in_features=None,
        out_features=None,
    ):
        super().__init__(model_path, dataset, from_file, model_name)
        self.in_features = in_features
        self.out_features = out_features

    def prepare_model(self):
        ...

    def load_model(self, model_path):
        ...

    def save_model(self, model_path):
        raise NotImplementedError

    def save_to_onnx(self, model_path):
        return NotImplementedError

    def run_inference(self, X):
        raise NotImplementedError

    def get_framework_and_version(self):
        return "emlearn", "dummy"

    @classmethod
    def get_output_formats(cls):
        return ["emlearn"]

    def train_model(self):
        raise NotImplementedError

    def get_io_specification_from_model(self):
        return {
            "input": [
                {
                    "name": "input_1",
                    "shape": [1, self.in_features],
                    "dtype": "float64",
                }
            ],
            "processed_input": [
                {
                    "name": "processed_input_1",
                    "shape": [1, self.in_features],
                    "dtype": "float32",
                    "order": 0,
                }
            ],
            "output": [
                {
                    "name": "output_1",
                    "shape": [1, self.out_features],
                    "dtype": "float32",
                    "order": 0,
                }
            ],
            "processed_output": [
                {
                    "name": "class",
                    "shape": [1],
                    "dtype": "int64",
                }
            ],
        }

    @classmethod
    def derive_io_spec_from_json_params(cls, json_dict):
        raise NotImplementedError

    def convert_input_to_bytes(self, inputdata):
        data = bytes()
        for inp in inputdata:
            for x in inp:
                data += x.tobytes()
        return data

    def convert_output_from_bytes(self, outputdata: bytes) -> list[Any]:
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

    def postprocess_outputs(self, y):
        Y = np.argmax(np.asarray(y, dtype=np.float32), axis=-1).reshape(-1)
        return [Y]
