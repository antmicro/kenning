# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains PyTorch model for MagicWand dataset.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from kenning.cli.command_template import TRAIN
from kenning.core.dataset import Dataset
from kenning.core.exceptions import TrainingParametersMissingError
from kenning.datasets.magic_wand_dataset import MagicWandDataset
from kenning.modelwrappers.frameworks.pytorch import PyTorchWrapper
from kenning.utils.resource_manager import PathOrURI


class PyTorchMagicWandModelWrapper(PyTorchWrapper):
    """
    Model wrapper for Magic Wand model.
    """

    default_dataset = MagicWandDataset
    arguments_structure = {
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
        "window_size": {
            "argparse_name": "--window-size",
            "description": "Number of sensor samples",
            "type": int,
            "default": 128,
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

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Dataset,
        from_file: bool = False,
        model_name: Optional[str] = None,
        window_size: int = 128,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        num_epochs: Optional[int] = None,
        logdir: Optional[Path] = None,
    ):
        super().__init__(model_path, dataset, from_file, model_name)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.logdir = logdir
        self.window_size = window_size
        if dataset is not None:
            self.class_names = self.dataset.get_class_names()
            self.numclasses = len(self.class_names)
            self.save_io_specification(self.model_path)

    @classmethod
    def _get_io_specification(
        cls, window_size, numclasses=-1, class_names=None, batch_size=1
    ):
        io_spec = {
            "input": [
                {
                    "name": "input_1",
                    "shape": [
                        (batch_size, window_size, 3),
                        (batch_size, window_size, 3, 1),
                    ],
                    "dtype": "float32",
                }
            ],
            "processed_input": [
                {
                    "name": "input_1",
                    "shape": (batch_size, 1, window_size, 3),
                    "dtype": "float32",
                }
            ],
            "output": [
                {
                    "name": "out_layer",
                    "shape": (batch_size, numclasses),
                    "dtype": "float32",
                }
            ],
        }
        if class_names is not None:
            io_spec["output"][0]["class_names"] = class_names
        return io_spec

    @classmethod
    def derive_io_spec_from_json_params(cls, json_dict):
        return cls._get_io_specification(json_dict["window_size"])

    def get_io_specification_from_model(self):
        if self.dataset:
            return self._get_io_specification(
                self.window_size,
                self.numclasses,
                self.class_names,
                self.dataset.batch_size,
            )

        return self._get_io_specification(
            self.window_size, self.numclasses, self.class_names
        )

    def preprocess_input(self, X: List[np.ndarray]) -> List[np.ndarray]:
        X = super().preprocess_input(X)
        if type(X) is not np.ndarray:
            X = np.array(X, "float32")
        if len(X.shape) == 5 and X.shape[-1] == 1:
            X = np.squeeze(X, axis=(-1,))
        return [X]

    def create_model_structure(self):
        import torch

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, (4, 3), padding="same"),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((3, 3)),
            torch.nn.Dropout(0.1),
            torch.nn.Conv2d(8, 16, (4, 1), padding="same"),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((3, 1)),
            torch.nn.Dropout(0.1),
            torch.nn.Flatten(),
            torch.nn.Linear(224, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4),
            torch.nn.Linear(4, self.numclasses),
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
                if isinstance(m, torch.nn.Linear) or isinstance(
                    m, torch.nn.Conv2d
                ):
                    torch.nn.init.xavier_uniform_(m.weight)
                    torch.nn.init.zeros_(m.bias)

            self.model.apply(weights_init)
            self.model_prepared = True
            self.save_model(self.model_path)
        self.model.to(self.device)

    def train_model(self):
        import torch
        from torch.utils.data import Dataset as TorchDataset

        if not self.batch_size:
            self.batch_size = self.dataset.batch_size

        missing_params = []
        if not self.learning_rate:
            missing_params.append("learning_rate")

        if not self.num_epochs:
            missing_params.append("num_epochs")

        if not self.logdir:
            missing_params.append("logdir")
        else:
            self.logdir.mkdir(exist_ok=True, parents=True)

        if missing_params:
            raise TrainingParametersMissingError(missing_params)

        self.prepare_model()
        Xt, Xv, Yt, Yv = self.dataset.train_test_split_representations(0.25)

        self.dataset.standardize = False

        class MagicWandDatasetPytorch(TorchDataset):
            def __init__(self, inputs, labels, dataset, model, dev):
                self.inputs = inputs
                self.labels = labels
                self.dataset = dataset
                self.model = model
                self.device = dev

            def __len__(self):
                return len(self.inputs)

            def __getitem__(self, idx):
                X = self.dataset.prepare_input_samples([self.inputs[idx]])[0][
                    0
                ]
                y = np.array(self.labels[idx])
                # Checking whether reordering axes is enough to convert data
                # point into the correct shape, or if we need to add an axis
                if len(X.shape) != len(
                    self.model.get_io_specification_from_model()["input"][0][
                        "shape"
                    ][0]
                ):
                    X = X[..., np.newaxis]
                X = torch.from_numpy(X.transpose((2, 0, 1)).astype("float32"))
                y = torch.from_numpy(y)
                return (X, y)

        mean, std = self.dataset.get_input_mean_std()

        traindat = MagicWandDatasetPytorch(
            Xt,
            Yt,
            self.dataset,
            self,
            self.device,
        )

        validdat = MagicWandDatasetPytorch(
            Xv,
            Yv,
            self.dataset,
            self,
            self.device,
        )

        trainloader = torch.utils.data.DataLoader(
            traindat, batch_size=self.batch_size, num_workers=0, shuffle=True
        )

        validloader = torch.utils.data.DataLoader(
            validdat, batch_size=self.batch_size, num_workers=0, shuffle=True
        )

        self.model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()
        import torch.optim as optim

        opt = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        def postprocess(
            outputs: Any, labels: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            return (
                torch.max(outputs.data, 1)[1],
                labels if labels.dim() == 1 else torch.max(labels, 1)[1],
            )

        self._train_model(
            trainloader, validloader, opt, criterion, postprocess
        )
