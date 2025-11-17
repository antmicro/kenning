# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains PyTorch model for the pet classification.

Pretrained on ImageNet dataset, trained on Pet Dataset.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from kenning.cli.command_template import TRAIN
from kenning.core.dataset import Dataset
from kenning.core.exceptions import TrainingParametersMissingError
from kenning.datasets.pet_dataset import PetDataset
from kenning.modelwrappers.frameworks.pytorch import PyTorchWrapper
from kenning.utils.resource_manager import PathOrURI


class PyTorchPetDatasetMobileNetV2(PyTorchWrapper):
    """
    Model wrapper for pet classification in PyTorch.
    """

    default_dataset = PetDataset
    pretrained_model_uri = "kenning:///models/classification/pytorch_pet_dataset_mobilenetv2_full_model.pth"
    arguments_structure = {
        "class_count": {
            "argparse_name": "--num-classes",
            "description": "Number of classes that the model can classify",
            "type": int,
            "default": 37,
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
        from_file: bool = True,
        model_name: Optional[str] = None,
        class_count: int = 37,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        num_epochs: Optional[int] = None,
        logdir: Optional[Path] = None,
        export_dict: bool = True,
    ):
        super().__init__(
            model_path, dataset, from_file, model_name, export_dict
        )
        self.class_count = class_count
        if hasattr(dataset, "numclasses"):
            self.numclasses = dataset.numclasses
        else:
            self.numclasses = class_count

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.logdir = logdir

    @classmethod
    def _get_io_specification(cls, numclasses, batch_size=1):
        return {
            "input": [
                {
                    "name": "input_1",
                    "shape": [
                        (batch_size, 3, 224, 224),
                        (batch_size, 224, 224, 3),
                    ],
                    "dtype": "float32",
                }
            ],
            "processed_input": [
                {
                    "name": "input_1",
                    "shape": (batch_size, 3, 224, 224),
                    "dtype": "float32",
                }
            ],
            "output": [
                {
                    "name": "548",
                    "shape": (batch_size, numclasses),
                    "dtype": "float32",
                }
            ],
        }

    @classmethod
    def derive_io_spec_from_json_params(cls, json_dict):
        return cls._get_io_specification(json_dict["class_count"])

    def get_io_specification_from_model(self):
        if self.dataset:
            if hasattr(self.dataset, "numclasses"):
                assert self.numclasses == self.dataset.numclasses
            return self._get_io_specification(
                self.numclasses, self.dataset.batch_size
            )

        return self._get_io_specification(self.numclasses)

    def preprocess_input(self, X: List[np.ndarray]) -> List[Any]:
        if np.ndim(X[0]) == 3:
            X = [np.expand_dims(X[0], 0)]

        if (
            self.dataset
            and getattr(self.dataset, "image_memory_layout", None) == "NCHW"
        ):
            return X
        else:
            return [x.transpose(0, 3, 1, 2) for x in X]

    def create_model_structure(self):
        from torchvision import models

        self.model = models.mobilenet_v2(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        import torch

        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(1280, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, self.numclasses),
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

            self.model.classifier.apply(weights_init)
            self.model_prepared = True
            self.save_model(self.model_path)
        self.model.to(self.device)

    def train_model(self):
        import torch
        from torch.utils.data import Dataset as TorchDataset
        from torchvision import transforms

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

        class PetDatasetPytorch(TorchDataset):
            def __init__(
                self, inputs, labels, dataset, model, dev, transform=None
            ):
                self.inputs = inputs
                self.labels = labels
                self.transform = transform
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
                X = torch.from_numpy(X.astype("float32"))
                y = torch.from_numpy(y)
                if self.transform:
                    X = self.transform(X)
                return (X, y)

        mean, std = self.dataset.get_input_mean_std()

        traindat = PetDatasetPytorch(
            Xt,
            Yt,
            self.dataset,
            self,
            self.device,
            transform=transforms.Compose(
                [
                    transforms.ColorJitter(0.1, 0.1),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean, std),
                ]
            ),
        )

        validdat = PetDatasetPytorch(
            Xv,
            Yv,
            self.dataset,
            self,
            self.device,
            transform=transforms.Compose(
                [
                    transforms.ColorJitter(0.1, 0.1),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean, std),
                ]
            ),
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
