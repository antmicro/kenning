# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides base methods for using PyTorch models in Kenning.
"""

import copy
from abc import ABC
from collections import OrderedDict
from typing import Any, List, Optional

import numpy as np

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.utils.resource_manager import PathOrURI


class PyTorchWrapper(ModelWrapper, ABC):
    """
    Base model wrapper for PyTorch models.
    """

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Dataset,
        from_file: bool = True,
        model_name: Optional[str] = None,
    ):
        super().__init__(model_path, dataset, from_file, model_name)
        self._device = None

    @property
    def device(self):
        import torch

        if self._device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        return self._device

    def load_weights(self, weights: OrderedDict):
        """
        Loads the model state using ``weights``.
        It assumes that the model structure is already defined.

        Parameters
        ----------
        weights : OrderedDict
            Dictionary used to load model weights.
        """
        self.model.load_state_dict(copy.deepcopy(weights))
        self.model.eval()

    def create_model_structure(self):
        """
        Recreates the model structure.
        Every PyTorchWrapper subclass has to implement its own architecture.
        """
        raise NotImplementedError

    def load_model(self, model_path: PathOrURI):
        import torch

        input_data = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )

        # If the file contains only the weights
        # we have to recreate the model's structure
        # Otherwise we just load the model
        if isinstance(input_data, OrderedDict):
            self.create_model_structure()
            self.load_weights(input_data)
        elif isinstance(input_data, torch.nn.Module):
            self.model = input_data

    def save_to_onnx(self, model_path: PathOrURI):
        import torch

        self.get_io_specification()
        input_spec = self.io_specification[
            "processed_input"
            if "processed_input" in self.io_specification
            else "input"
        ]

        self.prepare_model()
        x = tuple(
            torch.randn(
                [s if s > 0 else s for s in spec["shape"]], device="cpu"
            )
            for spec in input_spec
        )

        torch.onnx.export(
            self.model.to(device="cpu"),
            x,
            model_path,
            opset_version=11,
            input_names=[
                spec["name"] for spec in self.get_io_specification()["input"]
            ],
            output_names=[
                spec["name"] for spec in self.get_io_specification()["output"]
            ],
        )

    def save_model(self, model_path: PathOrURI, export_dict: bool = True):
        import torch

        self.prepare_model()
        if export_dict:
            torch.save(self.model.state_dict(), model_path)
        else:
            torch.save(self.model, model_path)

    def preprocess_input(self, X: List[np.ndarray]) -> List[Any]:
        import torch

        return [torch.Tensor(x).to(self.device) for x in X]

    def postprocess_outputs(self, y: List[Any]) -> List[np.ndarray]:
        import torch

        if isinstance(y, torch.Tensor):
            return y.detach().cpu().numpy()
        if isinstance(y, np.ndarray):
            return y
        if isinstance(y, list):
            return [self.postprocess_outputs(_y) for _y in y]
        raise NotImplementedError

    def run_inference(self, X: List[Any]) -> List[Any]:
        self.prepare_model()
        y = self.model(*X)
        if not isinstance(y, (list, tuple)):
            y = [y]
        return y

    def get_framework_and_version(self):
        import torch

        return ("torch", torch.__version__)

    @classmethod
    def get_output_formats(cls):
        return ["onnx", "torch"]

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

    def get_model_size(self) -> float:
        """
        Calculates model size, combining parameters and buffers.

        Returns
        -------
        float
            The model size in KB.
        """
        model_size = 0
        for param in self.model.parameters():
            model_size += param.nelement() * param.element_size()
        for buffer in self.model.buffers():
            model_size += buffer.nelement() * buffer.element_size()
        return model_size / 1024
