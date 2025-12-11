# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides base methods for using PyTorch models in Kenning.
"""

import copy
from abc import ABC
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Tuple, TypeVar

import numpy as np
from sklearn import metrics
from tqdm import tqdm

from kenning.converters import converter_registry
from kenning.core.dataset import Dataset
from kenning.core.exceptions import NotSupportedError
from kenning.core.model import ModelWrapper
from kenning.utils.logger import KLogger, LoggerProgressBar
from kenning.utils.resource_manager import PathOrURI

DataLoader = TypeVar("torch.utils.data.DataLoader")
Optimizer = TypeVar("torch.optim.Optimizer")
Tensor = TypeVar("torch.Tensor")


class PyTorchWrapper(ModelWrapper, ABC):
    """
    Base model wrapper for PyTorch models.
    """

    arguments_structure = {
        "export_dict": {
            "argparse_name": "--export-dict",
            "description": "If enabled, only the model state dictionary will be saved, not the whole model.",  # noqa: E501
            "type": bool,
            "default": True,
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Dataset,
        from_file: bool = True,
        model_name: Optional[str] = None,
        export_dict: bool = True,
    ):
        super().__init__(model_path, dataset, from_file, model_name)
        self.export_dict = export_dict
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
        if weights:
            self.model.load_state_dict(copy.deepcopy(weights))
        self.model.eval()

    def create_model_structure(self):
        """
        Recreates the model structure.
        Every PyTorchWrapper subclass has to implement its own architecture.
        """
        raise NotSupportedError

    def load_model(self, model_path: PathOrURI):
        import torch

        try:
            input_data = torch.load(
                self.model_path, map_location=self.device, weights_only=False
            )
        except FileNotFoundError:
            input_data = OrderedDict()

        # If the file contains only the weights
        # we have to recreate the model's structure
        # Otherwise we just load the model
        if isinstance(input_data, OrderedDict):
            self.create_model_structure()
            self.load_weights(input_data)
        elif isinstance(input_data, torch.nn.Module):
            self.model = input_data

    def save_to_onnx(self, model_path: PathOrURI):
        import tempfile
        from pathlib import Path

        import onnx
        import torch

        self.prepare_model()

        torch_path = Path(tempfile.NamedTemporaryFile().name)
        torch.save(self.model, torch_path)

        io_spec = self.get_io_specification()

        from copy import deepcopy

        io_spec = deepcopy(io_spec)
        io_spec["input"] = (
            io_spec["processed_input"]
            if "processed_input" in io_spec
            else io_spec["input"]
        )
        conversion_kwargs = {
            "io_spec": io_spec,
        }
        onnx_model = converter_registry.convert(
            torch_path, "torch", "onnx", **conversion_kwargs
        )
        onnx.save(onnx_model, model_path)

    def save_model(
        self, model_path: PathOrURI, export_dict: Optional[bool] = None
    ):
        import torch

        self.prepare_model()
        if export_dict is None:
            export_dict = self.export_dict
        if export_dict:
            torch.save(self.model.state_dict(), model_path)
        else:
            torch.save(self.model, model_path)

    def run_inference(self, X: List[Any]) -> List[Any]:
        import torch

        self.prepare_model()
        X = [torch.Tensor(x).to(self.device) for x in X]
        y = self.model(*X)
        if not isinstance(y, (list, tuple)):
            y = [y]
        y = [
            yy.detach().cpu().numpy() if isinstance(yy, torch.Tensor) else yy
            for yy in y
        ]
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

    def _train_model(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        opt: Optimizer,
        criterion: Callable[[Tensor], Tensor],
        postprocess: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]],
        metric_func: Callable[
            [np.ndarray, np.ndarray], float
        ] = metrics.accuracy_score,
        epoch_start_hook: Callable[[int], None] = None,
    ):
        """
        General training loop for PyTorch models.

        Parameters
        ----------
        train_loader : DataLoader
            The train set loader.
        test_loader : DataLoader
            The test set loader.
        opt : Optimizer
            The model optimizer.
        criterion : Callable[[Tensor], Tensor]
            The function calculation loss.
        postprocess : Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]
            The function post-processing the model output and label.
        metric_func : Callable[[np.ndarray, np.ndarray], float]
            The function calculation metric.
        epoch_start_hook : Callable[[int], None]
            The hook called at the beginning of each epoch.
        """
        import torch
        from torch.utils.tensorboard import SummaryWriter

        best_metric = 0

        writer = SummaryWriter(log_dir=self.logdir)

        for epoch in range(self.num_epochs):
            if epoch_start_hook:
                epoch_start_hook(epoch)

            self.model.train()
            loss_sum = torch.zeros(1).to(self.device)
            loss_count = 0
            with LoggerProgressBar() as logger_progress_bar:
                bar = tqdm(train_loader, **logger_progress_bar.kwargs)
                for input, labels in bar:
                    input = input.to(self.device)
                    labels = labels.to(self.device)
                    opt.zero_grad()

                    outputs = self.model(input)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    opt.step()

                    loss_sum += loss
                    loss_count += 1
                    bar.set_description(
                        f"train epoch: {epoch:3d} loss: "
                        f"{loss_sum.data.cpu().numpy().sum() / loss_count:.3f}"
                    )

            writer.add_scalar(
                "Loss/train", loss_sum.data.cpu().numpy() / loss_count, epoch
            )

            self.model.eval()
            predicted = np.array([])
            true = np.array([])
            with torch.no_grad(), LoggerProgressBar() as logger_progress_bar:
                bar = tqdm(test_loader, **logger_progress_bar.kwargs)
                for input, labels in bar:
                    input = input.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(input)
                    outputs, labels = postprocess(outputs, labels)
                    predicted = np.concatenate(
                        [predicted, outputs.cpu().numpy()]
                    )
                    true = np.concatenate((true, labels.cpu().numpy()))
                    metric = metric_func(true, predicted)
                    bar.set_description(
                        f"valid epoch: {epoch:3d} {str(metric_func)}: "
                        f"{metric:.3f}"
                    )

                writer.add_scalar(f"{str(metric_func)}/valid", metric, epoch)

                if metric > best_metric:
                    self.save_model(self.model_path)
                    best_metric = metric
        KLogger.info(f"Model training finished, best score: {best_metric}")
        self.save_model(
            self.model_path.with_stem(f"{self.model_path.stem}_final"),
        )

        self.dataset.standardize = True

        writer.close()
        self.model.eval()
