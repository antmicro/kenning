# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
Implements an abstract anomaly detection class.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from tqdm import tqdm

from kenning.cli.command_template import TRAIN
from kenning.core.exceptions import TrainingParametersMissingError
from kenning.datasets.anomaly_detection_dataset import AnomalyDetectionDataset
from kenning.modelwrappers.frameworks.pytorch import PyTorchWrapper
from kenning.utils.logger import KLogger, LoggerProgressBar
from kenning.utils.resource_manager import PathOrURI


class PyTorchAnomalyDetectionWrapper(PyTorchWrapper, ABC):
    """
    Base model wrapper for PyTorch Anomaly Detection models.
    """

    arguments_structure = {
        "batch_size": {
            "argparse_name": "--batch-size",
            "description": "Batch size for training. If not assigned, dataset batch size will be used",  # noqa: E501
            "type": int,
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
        "clean_trainset": {
            "argparse_name": "--clean-trainset",
            "description": "Force the model to train only on samples labeled as normal",  # noqa: E501
            "type": bool,
            "default": True,
            "subcommands": [TRAIN],
        },
        "metric": {
            "argparse_name": "--metric",
            "description": "The metric to use for saving the best model",
            "type": str,
            "default": "f1",
            "subcommands": [TRAIN],
        },
        "learning_rate": {
            "argparse_name": "--learning-rate",
            "description": "Set the learning rate",
            "type": float,
            "default": 1e-3,
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
        dataset: AnomalyDetectionDataset,
        num_epochs: int,
        clean_trainset: bool = True,
        metric: str = "f1",
        learning_rate: float = 1e-3,
        from_file: bool = True,
        model_name: Optional[str] = None,
        logdir: Optional[Path] = None,
        export_dict: bool = True,
    ):
        super().__init__(model_path, dataset, from_file, model_name)
        self.num_epochs = num_epochs
        self.clean_trainset = clean_trainset
        self.learning_rate = learning_rate
        self.logdir = logdir

        if dataset and hasattr(dataset, "get_input_mean_std"):
            self.mean, self.std = self.dataset.get_input_mean_std()
        else:
            self.mean = None
            self.std = None
        self.metric = metric

    def prepare_model(self):
        if self.model_prepared:
            return None

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
        X = (X - self.mean) / self.std
        X = np.expand_dims(X, -3)
        return [X]

    @abstractmethod
    def prepare_criterion(self) -> nn.Module:
        ...

    @abstractmethod
    def prepare_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        ...

    @abstractmethod
    def _forward_pass(
        self,
        criterion: nn.Module,
        inputs: Any,
        labels: Any,
        is_validation: bool = False,
    ) -> (torch.Tensor, Any):
        """
        Perform a forward pass operations.

        Parameters
        ----------
        criterion: nn.Module
            An instantiated loss function to use.
        inputs: Any
            Input to the model ``self.model(inputs)``.
        labels: Any
            True value given the input.
        is_validation: bool
            True if the function is called from the validation phase.

        Returns
        -------
        (torch.Tensor, Any)
            Returns the output of the loss function alongside
            the predicted label.
        """
        ...

    def _prepare_training(self):
        if not self.batch_size:
            self.batch_size = self.dataset.batch_size

        missing_params = []
        if not self.learning_rate:
            missing_params.append("learning_rate")

        if not self.num_epochs:
            missing_params.append("num_epochs")

        if missing_params:
            raise TrainingParametersMissingError(missing_params)

        train_loader, test_loader = self._get_torch_dataset()

        self.model.to(self.device)
        criterion = self.prepare_criterion()
        opt = self.prepare_optimizer(self.model)
        return train_loader, test_loader, criterion, opt

    def _get_torch_dataset(self):
        from torch.utils.data import Dataset as TorchDataset

        (
            Xtr,
            Xte,
            Ytr,
            Yte,
        ) = self.dataset.train_test_split_representations()

        class AnomalyDatasetTorch(TorchDataset):
            def __init__(
                self,
                inputs,
                labels,
                dataset,
                dev,
                transform=lambda x: x,
                clean=False,
            ):
                self.inputs = inputs
                self.labels = labels
                self.dataset = dataset
                self.device = dev
                self.transform = transform
                if clean:
                    self.inputs, self.labels = self._clean_dataset(
                        inputs, labels
                    )

            def _clean_dataset(self, inputs, labels):
                if self.dataset.label_type == "per_timestep":
                    mask = [not label.any() for label in labels]
                else:
                    mask = [label != 1 for label in labels]

                cleaned_inputs = [
                    inputs[i] for i in range(len(inputs)) if mask[i]
                ]
                cleaned_labels = [
                    labels[i] for i in range(len(labels)) if mask[i]
                ]
                return cleaned_inputs, cleaned_labels

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
                # Get last timestep.
                # print(f'y: {y[0][-1]}')
                y = torch.from_numpy(
                    np.asarray(
                        y[0][-1]
                        if self.dataset.label_type == "per_timestep"
                        else y[0]
                    )
                )
                return (X.squeeze(), y)

        trainset = AnomalyDatasetTorch(
            Xtr,
            Ytr,
            self.dataset,
            self.device,
            clean=self.clean_trainset,
            transform=lambda x: self.preprocess_input(x)[0],
        )

        testset = AnomalyDatasetTorch(
            Xte,
            Yte,
            self.dataset,
            self.device,
            clean=False,
            transform=lambda x: self.preprocess_input(x)[0],
        )

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True,
        )

        test_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )

        return train_loader, test_loader

    def train_model(self):
        (
            train_loader,
            test_loader,
            criterion,
            opt,
        ) = self._prepare_training()

        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=self.logdir)

        best_metric = 0.0

        for epoch in range(self.num_epochs):
            self.model.train()

            loss_sum = torch.zeros(1).to(self.device)
            loss_count = 0

            with LoggerProgressBar() as logger_progress_bar:
                bar = tqdm(train_loader, **logger_progress_bar.kwargs)
                for input, labels in bar:
                    input = input.to(self.device)
                    labels = labels.to(self.device)
                    opt.zero_grad()

                    loss, _ = self._forward_pass(criterion, input, labels)
                    loss.backward()
                    opt.step()

                    loss_sum += loss
                    loss_count += 1

                    bar.set_description(
                        f"training epoch: {epoch:3d} loss: "
                        f"{loss_sum.data.cpu().numpy().sum() / loss_count:.5f}"
                    )

            writer.add_scalar(
                "Loss/train", loss_sum.data.cpu().numpy() / loss_count, epoch
            )

            self.model.eval()

            loss_sum = torch.zeros(1).to(self.device)
            loss_count = 0

            with torch.no_grad(), LoggerProgressBar() as logger_progress_bar:
                predicted = np.array([])
                true = np.array([])

                bar = tqdm(test_loader, **logger_progress_bar.kwargs)
                for input, labels in bar:
                    input = input.to(self.device)
                    labels = labels.to(self.device)
                    loss, pred = self._forward_pass(
                        criterion, input, labels, is_validation=True
                    )
                    true = np.concatenate((true, labels.cpu().numpy()))
                    predicted = np.concatenate((predicted, pred.cpu().numpy()))

                    loss_sum += loss
                    loss_count += 1

                    bar.set_description(
                        f"evaluation epoch: {epoch:3d} loss: "
                        f"{loss_sum.data.cpu().numpy().sum() / loss_count:.5f}"
                    )

                # Calculate other metrics.
                metrics, threshold = self._calculate_metrics(true, predicted)
                for k, v in metrics.items():
                    writer.add_scalar(f"{k}/valid", v, epoch)
                    KLogger.info(f"- {k}: {v:.5f}")

                KLogger.info(f"Metric = {self.metric}")
                if metrics[self.metric] > best_metric:
                    self.save_model(self.model_path)
                    best_metric = metrics[self.metric]
                    self.threshold = threshold

        KLogger.info(
            f"Model training finished, best {self.metric} score: {best_metric}"
        )
        self.save_model(
            self.model_path.with_stem(f"{self.model_path.stem}_final")
        )

        self.dataset.standardize = True
        writer.close()
        self.model.eval()

    def _calculate_metrics(
        self, true: np.ndarray, predicted: np.ndarray
    ) -> (Dict[str, float], float):
        """
        Calculate the metrics given the true and predicted arrays.

        Parameters
        ----------
        true: np.ndarray
            The array containing the true values.
        predicted: np.ndarray
            The array containing predicted classes.

        Returns
        -------
        (Dict[str, float], float)
            Dictionary of calculate metrics. The second return value is the
            threshold used to calculate the metrics.

        """
        # By default, we only have the f1 score.
        f1 = metrics.f1_score(true, predicted)
        return {
            "f1": f1,
        }
