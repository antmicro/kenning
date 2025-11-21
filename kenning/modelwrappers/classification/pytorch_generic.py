# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing a generic PyTorch classification model wrapper.
"""

import importlib.util
import sys
from functools import partial

import numpy as np

from kenning.core import metrics
from kenning.core.exceptions import TrainingParametersMissingError
from kenning.datasets.tabular_dataset import TabularDataset
from kenning.modelwrappers.frameworks.pytorch import PyTorchWrapper


class PyTorchGenericClassification(PyTorchWrapper):
    """
    Wrapper for generic PyTorch classfication models.
    """

    arguments_structure = {
        "batch_size": {
            "argparse_name": "--training-batch-size",
            "description": "The batch size for providing the input data",
            "type": int,
            "default": 1,
        },
        "model_source": {
            "argparse_name": "--model-source",
            "description": "Python script with functions for model generation",
            "type": str,
            "nullable": True,
            "default": None,
        },
        "learning_rate": {
            "argparse_name": "--learning-rate",
            "description": "Learning rate",
            "type": float,
            "default": 0.01,
        },
        "num_epochs": {
            "argparse_name": "--num-epochs",
            "description": "Number of epochs",
            "type": int,
            "default": 10,
        },
    }

    DEFAULT_SAVE_MODEL_EXPORT_DICT = True

    default_dataset = TabularDataset

    def __init__(
        self,
        model_path,
        dataset,
        batch_size=1,
        from_file=True,
        model_name=None,
        model_source=None,
        learning_rate=None,
        num_epochs=None,
    ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.logdir = "logdir"
        self.batch_size = batch_size
        self.model_source = model_source
        super().__init__(model_path, dataset, from_file, model_name)

    def prepare_model(self):
        if self.model_prepared:
            return None

        self.from_file = True

        if self.from_file:
            self.load_model(self.model_path)
            self.model_prepared = True
        else:
            msg = "'from_file' is false"
            raise RuntimeError(msg)

        self.model.to(self.device)

    def create_model_structure(self):
        if self.model_source is None:
            return

        spec = importlib.util.spec_from_file_location(
            "model", self.model_source
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module] = module
        spec.loader.exec_module(module)

        self.model = module.get_model()
        self.extra_models = (
            module.get_extra_models()
            if hasattr(module, "get_extra_models")
            else ()
        )

    def train_model(self):
        import torch
        import torch.optim as optim

        train_loader, test_loader = self._prepare_loaders()

        criterion = torch.nn.CrossEntropyLoss()
        opt = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        postprocess = self._prepare_postprocess()

        self.model.to(self.device)

        self._train_model(
            train_loader,
            test_loader,
            opt,
            criterion,
            postprocess,
            partial(metrics.f1_score, average="binary", pos_label=1),
        )

    def _prepare_missing_params(self):
        if not self.batch_size:
            self.batch_size = self.dataset.batch_size

        missing_params = []
        if not self.learning_rate:
            missing_params.append("learning_rate")

        if not self.num_epochs:
            missing_params.append("num_epochs")

        if missing_params:
            raise TrainingParametersMissingError(missing_params)

    def _prepare_postprocess(self):
        import torch

        def postprocess(
            outputs: torch.Tensor, labels: torch.Tensor
        ) -> torch.Tensor:
            return torch.argmax(outputs, dim=-1).flatten(), torch.argmax(
                labels, dim=-1
            ).flatten()

        return postprocess

    def _prepare_loaders(self, select_class=None):
        import torch
        from torch.utils.data import Dataset as TorchDataset

        (
            Xtr,
            Xte,
            Ytr,
            Yte,
        ) = self.dataset.train_test_split_representations()

        if select_class is not None:
            Xtr = [x for (x, y) in zip(Xtr, Ytr) if y == select_class]
            Ytr = [select_class] * len(Xtr)
            Xte = [x for (x, y) in zip(Xte, Yte) if y == select_class]
            Yte = [select_class] * len(Xte)

        class _TabularTorchDataset(TorchDataset):
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
                X = self.dataset.prepare_input_samples(X)
                y = self.dataset.prepare_output_samples(y)[0]

                if self.transform:
                    X = self.transform(X)
                X = torch.from_numpy(X)
                y = torch.from_numpy(np.asarray(y))
                return (X, y)

        train_set = _TabularTorchDataset(
            Xtr,
            Ytr,
            self.dataset,
            self.device,
            transform=lambda x: self.preprocess_input(x)[0],
        )
        test_set = _TabularTorchDataset(
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

        return train_loader, test_loader

    def get_io_specification_from_model(self):
        return self.get_io_specification_from_dataset(self.dataset)

    def get_io_specification_from_dataset(
        cls,
        dataset: TabularDataset,
    ):
        return {
            "input": [
                {
                    "name": "input_1",
                    "shape": (
                        dataset.batch_size,
                        dataset.num_features,
                    )
                    if dataset.window_size is None
                    else (
                        dataset.batch_size,
                        dataset.window_size,
                        dataset.num_features,
                    ),
                    "dtype": "float64",
                }
            ],
            "processed_input": [
                {
                    "name": "input_1",
                    "shape": (
                        dataset.batch_size,
                        dataset.num_features,
                    )
                    if dataset.window_size is None
                    else (
                        dataset.batch_size,
                        dataset.window_size,
                        dataset.num_features,
                    ),
                    "dtype": "float32",
                }
            ],
            "output": [
                {
                    "name": "probs",
                    "shape": (dataset.batch_size, len(dataset.class_names)),
                    "dtype": "float32",
                }
            ],
            "processed_output": [
                {
                    "name": "class",
                    "shape": (dataset.batch_size,),
                    "dtype": "int64",
                }
            ],
        }

    def preprocess_input(self, X):
        X = np.asarray(X[0], dtype=np.float32)
        return [X]

    def postprocess_outputs(self, y):
        Y = np.argmax(np.asarray(y, dtype=np.float32), axis=-1).reshape(-1)
        return [Y]

    def run_inference(self, X):
        import torch

        self.prepare_model()
        y = self.model(
            *[
                torch.tensor(x, device=self.device, dtype=torch.float32)
                for x in X
            ]
        )
        if not isinstance(y, (list, tuple)):
            y = [y]
        return [_y.detach().cpu().numpy() for _y in y]

    @classmethod
    def derive_io_spec_from_json_params(cls, json_dict):
        cls.get_io_specification(-1, -1)
