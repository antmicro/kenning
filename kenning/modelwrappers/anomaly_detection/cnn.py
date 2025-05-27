# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains simple Convolution Neural Network (CNN) model wrapper.

Compatible with AnomalyDetectionDataset.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from sklearn import metrics

from kenning.automl.auto_pytorch import AutoPyTorchModel
from kenning.cli.command_template import TRAIN
from kenning.core.model import TrainingParametersMissingError
from kenning.core.platform import Platform
from kenning.datasets.anomaly_detection_dataset import AnomalyDetectionDataset
from kenning.modelwrappers.frameworks.pytorch import PyTorchWrapper
from kenning.utils.resource_manager import PathOrURI


class PyTorchAnomalyDetectionCNN(PyTorchWrapper, AutoPyTorchModel):
    """
    Model wrapper for anomaly detection with CNN.

    It is compatible with AutoML flow.
    """

    DEFAULT_SAVE_MODEL_EXPORT_DICT = False

    default_dataset = AnomalyDetectionDataset
    arguments_structure = {
        "filters": {
            "description": "The list with number of filters of convolution layers",  # noqa: E501
            "type": list[int],
            "default": [8, 16],
            "AutoML": True,
            "list_range": (1, 6),
            "item_range": (2, 64),
        },
        "kernel_size": {
            "description": "The kernel size, applies to all convolution layers",  # noqa: E501
            "type": int,
            "enum": [1, 3],
            "default": 3,
            "AutoML": True,
        },
        "conv_padding": {
            "description": "The padding, applies to all convolution layers",
            "type": int,
            "default": 0,
            "AutoML": True,
            "item_range": (0, 2),
        },
        "conv_stride": {
            "description": "The stride, applies to all convolution layers",
            "type": int,
            "default": 1,
            "AutoML": True,
            "item_range": (1, 4),
        },
        "conv_dilation": {
            "description": "The dialtion, applies to all convolution layers",
            "type": int,
            "default": 1,
            "AutoML": True,
            "item_range": (1, 3),
        },
        "conv_activation": {
            "description": "The activation applied after each convolution layers",  # noqa: E501
            "type": str,
            "enum": ["ReLU", "Abs"],
            "nullable": True,
            "default": None,
            "AutoML": True,
        },
        "conv_batch_norm": {
            "description": "The batch normalization applied after each convolution layers",  # noqa: E501
            "type": str,
            "enum": ["Affine", "NoAffine"],
            "nullable": True,
            "default": None,
            "AutoML": True,
        },
        "pooling": {
            "description": "Whether to use pooling layer after the last convolution layer",  # noqa: E501
            "type": str,
            "enum": ["Max", "Avg"],
            "nullable": True,
            "default": None,
            "AutoML": True,
        },
        "pool_size": {
            "description": "The kernel size of pooling layer",
            "type": int,
            "default": 1,
            "AutoML": True,
            "item_range": (1, 3),
        },
        "pool_stride": {
            "description": "The stride of pooling layer",
            "type": int,
            "default": 1,
            "AutoML": True,
            "item_range": (1, 4),
        },
        "pool_dilation": {
            "description": "The dilation of pooling layer",
            "type": int,
            "default": 1,
            "AutoML": True,
            "item_range": (1, 3),
        },
        "fc_neurons": {
            "description": "The list with number of neurons for hidden fully connected layers",  # noqa: E501
            "type": list[int],
            "default": [16, 8, 4],
            "AutoML": True,
            "list_range": (0, 8),
            "item_range": (2, 64),
        },
        "batch_size": {
            "argparse_name": "--batch-size",
            "description": "Batch size for training. If not assigned, dataset batch size will be used.",  # noqa: E501
            "type": int,
            "default": None,
            "subcommands": [TRAIN],
        },
        "learning_rate": {
            "description": "Learning rate for training",
            "type": float,
            "default": None,
            "subcommands": [TRAIN],
        },
        "num_epochs": {
            "argparse_name": "--num-epochs",
            "description": "Number of epochs to train for",
            "type": int,
            "default": None,
            "subcommands": [TRAIN],
        },
        "evaluate": {
            "argparse_name": "--evaluate",
            "description": "True if the model should be evaluated each epoch",
            "type": bool,
            "default": True,
            "subcommands": [TRAIN],
        },
        "logdir": {
            "argparse_name": "--logdir",
            "description": "Path to the logging directory",
            "type": Path,
            "default": None,
            "subcommands": [TRAIN],
        },
    }
    model_class = "kenning.modelwrappers.anomaly_detection.models.cnn.AnomalyDetectionCNN"  # noqa: E501

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: AnomalyDetectionDataset,
        from_file: bool = True,
        model_name: Optional[str] = None,
        filters: List[int] = [8, 16],
        kernel_size: int = 3,
        conv_padding: int = 0,
        conv_stride: int = 1,
        conv_dilation: int = 1,
        conv_activation: Optional[Literal["ReLU", "Abs"]] = None,
        conv_batch_norm: Optional[Literal["Affine", "NoAffine"]] = None,
        pooling: Optional[Literal["Max", "Avg"]] = None,
        pool_size: int = 1,
        pool_stride: int = 2,
        pool_dilation: int = 1,
        fc_neurons: List[int] = [8],
        fc_activation: Optional[Literal["ReLU", "Abs"]] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        num_epochs: Optional[int] = None,
        evaluate: bool = True,
        logdir: Optional[Path] = None,
    ):
        super().__init__(model_path, dataset, from_file, model_name)

        self.filters = filters
        self.kernel_size = kernel_size
        self.conv_padding = conv_padding
        self.conv_stride = conv_stride
        self.conv_dilation = conv_dilation
        self.conv_activation = conv_activation
        self.conv_batch_norm = conv_batch_norm

        self.pooling = pooling
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.pool_dilation = pool_dilation

        self.fc_neurons = fc_neurons
        self.fc_activation = fc_activation

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.evaluate = evaluate
        self.logdir = logdir

        if dataset:
            self.mean, self.std = self.dataset.get_input_mean_std()

    @classmethod
    def model_params_from_context(
        cls,
        dataset: AnomalyDetectionDataset,
        platform: Optional[Platform] = None,
    ):
        return {
            "window_size": dataset.window_size,
            "feature_size": dataset.num_features,
        }

    def create_model_structure(self, **kwargs):
        from kenning.modelwrappers.anomaly_detection.models.cnn import (
            AnomalyDetectionCNN,
        )

        self.model = AnomalyDetectionCNN(
            filters=self.filters,
            kernel_size=self.kernel_size,
            conv_padding=self.conv_padding,
            conv_stride=self.conv_stride,
            conv_dilation=self.conv_dilation,
            conv_activation=self.conv_activation,
            conv_batch_norm=self.conv_batch_norm,
            pooling=self.pooling,
            pool_size=self.pool_size,
            pool_stride=self.pool_stride,
            pool_dilation=self.pool_dilation,
            fc_neurons=self.fc_neurons,
            fc_activation=self.fc_activation,
            input_shape=None,
            **kwargs,
            **PyTorchAnomalyDetectionCNN.model_params_from_context(
                self.dataset
            ),
        )

    def prepare_model(self):
        if self.model_prepared:
            return None
        import torch

        if self.from_file:
            self.load_model(self.model_path)
            self.model_prepared = True
        else:
            self.create_model_structure()

            def weights_init(m):
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    torch.nn.init.zeros_(m.bias)

            self.model.apply(weights_init)
            self.model_prepared = True
            self.save_model(self.model_path)
        self.model.to(self.device)

    def preprocess_input(self, X) -> List[Any]:
        X = np.asarray(X[0])
        X = (X - self.mean) / (3 * self.std)
        X = np.expand_dims(X, -3)
        return [X]

    def postprocess_outputs(self, y: List[Any]) -> List[np.ndarray]:
        # Get index of bigger value - simplified softmax
        anomalies = (
            np.argmax(np.asarray(y), axis=-1).reshape(-1).astype(np.int8)
        )
        return (anomalies,)

    def run_inference(self, X: List[Any]) -> List[Any]:
        import torch

        self.prepare_model()
        y = self.model(*[torch.tensor(x, device=self.device) for x in X])
        if not isinstance(y, (list, tuple)):
            y = [y]
        return [_y.detach().cpu().numpy() for _y in y]

    def _prepare_training(self):
        import torch
        import torch.optim as optim
        from torch.utils.data import Dataset as TorchDataset

        if not self.batch_size:
            self.batch_size = self.dataset.batch_size

        missing_params = []
        if not self.learning_rate:
            missing_params.append("learning_rate")

        if not self.num_epochs:
            missing_params.append("num_epochs")

        if missing_params:
            raise TrainingParametersMissingError(missing_params)

        (
            Xtr,
            Xte,
            Ytr,
            Yte,
        ) = self.dataset.train_test_split_representations()

        class AnomalyDatasetPytorch(TorchDataset):
            def __init__(
                self,
                inputs,
                labels,
                dataset,
                dev,
                transform=None,
            ):
                self.inputs = inputs
                self.labels = labels
                self.dataset = dataset
                self.device = dev
                self.transform = transform

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                X = self.inputs[idx]
                y = self.labels[idx]
                X = self.dataset.prepare_input_samples(X)[0]
                y = self.dataset.prepare_output_samples(y)
                if self.transform:
                    X = self.transform(X)
                X = torch.from_numpy(X)
                y = torch.from_numpy(np.asarray(y[0]))
                return (X, y)

        train_set = AnomalyDatasetPytorch(
            Xtr,
            Ytr,
            self.dataset,
            self.device,
            transform=lambda x: self.preprocess_input(x)[0],
        )
        test_set = AnomalyDatasetPytorch(
            Xte,
            Yte,
            self.dataset,
            self.device,
            transform=lambda x: self.preprocess_input(x)[0],
        )

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, num_workers=0, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self.batch_size, num_workers=0, shuffle=True
        )

        self.model.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        opt = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        def postprocess(
            outputs: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor:
            return torch.argmax(outputs, dim=-1).flatten(), labels

        return train_loader, test_loader, criterion, opt, postprocess

    def train_model(self):
        (
            train_loader,
            test_loader,
            criterion,
            opt,
            postprocess,
        ) = self._prepare_training()

        self._train_model(
            train_loader,
            test_loader,
            opt,
            criterion,
            postprocess,
            metrics.f1_score,
        )

    def get_io_specification_from_model(self) -> Dict[str, List[Dict]]:
        return self.get_io_specification_from_dataset(self.dataset)

    def get_io_specification_from_dataset(
        cls,
        dataset,
    ) -> Dict[str, List[Dict]]:
        return cls._get_io_specification(
            dataset.num_features,
            dataset.window_size,
            dataset.batch_size,
        )

    @classmethod
    def derive_io_spec_from_json_params(
        cls, json_dict: Dict
    ) -> Dict[str, List[Dict]]:
        cls.get_io_specification(-1, -1)

    @classmethod
    def _get_io_specification(
        cls,
        num_features,
        window_size,
        batch_size: int = -1,
    ) -> Dict[str, List[Dict]]:
        return {
            "input": [
                {
                    "name": "input_1",
                    "shape": (
                        batch_size,
                        window_size,
                        num_features,
                    ),
                    "dtype": "float32",
                }
            ],
            "processed_input": [
                {
                    "name": "input_1",
                    "shape": (
                        batch_size,
                        1,
                        window_size,
                        num_features,
                    ),
                    "dtype": "float32",
                }
            ],
            "output": [
                {
                    "name": "distances",
                    "shape": (batch_size, 2),
                    "dtype": "float32",
                }
            ],
            "processed_output": [
                {
                    "name": "anomalies",
                    "shape": (batch_size,),
                    "dtype": "int8",
                }
            ],
        }

    @classmethod
    def define_forbidden_clauses(cls, cs, **kwargs):
        import ConfigSpace as CS

        cs = super().define_forbidden_clauses(cs, **kwargs)

        model_component = cls.get_component_name()
        network_back = cs.get_hyperparameter("network_backbone:__choice__")
        network_back = CS.ForbiddenEqualsClause(
            network_back,
            model_component,
        )

        # Disable SparseInit as it is only available for 2D weights
        network_init_param = cs.get_hyperparameter("network_init:__choice__")
        sparse_clause = CS.ForbiddenInClause(
            network_init_param, ("SparseInit",)
        )

        cs.add_forbidden_clause(
            CS.ForbiddenAndConjunction(
                network_back,
                sparse_clause,
            )
        )
        return cs
