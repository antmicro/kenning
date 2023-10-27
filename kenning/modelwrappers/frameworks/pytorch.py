# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides base methods for using PyTorch models in Kenning.
"""

import copy
from abc import ABC
from typing import Optional
from collections import OrderedDict

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
        import torch

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # noqa: E501

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

        input_data = torch.load(self.model_path, map_location=self.device)

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

        self.prepare_model()
        x = tuple(
            torch.randn(spec["shape"], device="cpu")
            for spec in self.get_io_specification()["input"]
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

    def preprocess_input(self, X):
        import torch

        return torch.Tensor(np.array(X)).to(self.device)

    def postprocess_outputs(self, y):
        import torch

        if isinstance(y, torch.Tensor):
            return y.detach().cpu().numpy()
        if isinstance(y, np.ndarray):
            return y
        if isinstance(y, list):
            return [self.postprocess_outputs(_y) for _y in y]
        raise NotImplementedError

    def run_inference(self, X):
        self.prepare_model()
        return self.model(X)

    def get_framework_and_version(self):
        import torch

        return ("torch", torch.__version__)

    def get_output_formats(self):
        return ["onnx", "torch"]
