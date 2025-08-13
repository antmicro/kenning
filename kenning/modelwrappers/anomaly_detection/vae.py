# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains Variational Autoencoder (VAE) model wrapper.

Compatible with AnomalyDetectionDataset.
"""

from enum import Enum
from random import shuffle
from typing import Any, Dict, List, Optional, Type

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

from kenning.automl.auto_pytorch import AutoPyTorchModel
from kenning.cli.command_template import TRAIN
from kenning.core.exceptions import TrainingParametersMissingError
from kenning.core.platform import Platform
from kenning.datasets.anomaly_detection_dataset import AnomalyDetectionDataset
from kenning.modelwrappers.frameworks.pytorch import PyTorchWrapper
from kenning.utils.logger import KLogger, LoggerProgressBar
from kenning.utils.resource_manager import PathOrURI

AVAILABLE_ACTIVATION_NAMES = [
    "elu",
    "leaky_relu",
    "relu",
    "sigmoid",
    "softmax",
    "softplus",
    "tanh",
]


class TrainerParams(str, Enum):
    """
    Parameters for VAETrainer.
    """

    BETA = "loss_beta"
    CAPACITY = "loss_capacity"
    CLIP_GRAD = "clip_grad_max_norm"


class PyTorchAnomalyDetectionVAE(PyTorchWrapper, AutoPyTorchModel):
    """
    Model wrapper for anomaly detection with VAE.

    The loss function used for this model contains:
    * MSE of received and inferred data,
    * KL divergence of generated distribution - its maximum value and scaling
    can be adjusted with capacity and beta.

    More information can be found in Burges et al
    'Understanding disentangling in beta-VAE'
    <https://arxiv.org/pdf/1804.03599.pdf>
    """

    DEFAULT_SAVE_MODEL_EXPORT_DICT = False

    default_dataset = AnomalyDetectionDataset
    arguments_structure = {
        "encoder_neuron_list": {
            "description": "List of dense layer dimensions of encoder",
            "type": list[int],
            "default": [16, 8],
            "AutoML": True,
            "list_range": (2, 6),
            "item_range": (4, 48),
        },
        "decoder_neuron_list": {
            "description": "List of dense layer dimensions of decoder",
            "type": list[int],
            "default": [16, 32],
            "AutoML": True,
            "list_range": (2, 6),
            "item_range": (4, 48),
        },
        "latent_dim": {
            "argparse_name": "--latent-dim",
            "description": "Dimensions of latent layer",
            "type": int,
            "default": 2,
            "AutoML": True,
            "item_range": (2, 48),
        },
        "hidden_activation": {
            "argparse_name": "--hidden-activation",
            "description": "Name of the activation of hidden layers",
            "type": str,
            "enum": AVAILABLE_ACTIVATION_NAMES,
            "default": "relu",
            "AutoML": True,
        },
        "output_activation": {
            "argparse_name": "--output-activation",
            "description": "Name of the activation of output layers",
            "type": str,
            "enum": ["sigmoid", "tanh"],
            "default": "sigmoid",
            "AutoML": True,
        },
        "batch_norm": {
            "argparse_name": "--batch-norm",
            "description": "Whether batch norm should be enabled",
            "type": bool,
            "default": False,
            "AutoML": True,
        },
        "dropout_rate": {
            "argparse_name": "--dropout-rate",
            "description": "Dropout rate - disabled if set to zero",
            "type": float,
            "default": 0.0,
            "AutoML": True,
            "item_range": (0.0, 0.7),
        },
        TrainerParams.BETA.value: {
            "description": "Parameter of the VAE loss function - scales KL divegence",  # noqa: E501
            "type": float,
            "default": 1.0,
            "subcommands": [TRAIN],
        },
        TrainerParams.CAPACITY.value: {
            "description": "Parameter of the VAE loss function - defines upper limit of KL divegence",  # noqa: E501
            "type": float,
            "default": 0.0,
            "subcommands": [TRAIN],
        },
        TrainerParams.CLIP_GRAD.value: {
            "description": "Max norm for clipping gradients",
            "type": float,
            "default": 2.0,
            "subcommands": [TRAIN],
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
    }

    model_class = "kenning.modelwrappers.anomaly_detection.models.vae.AnomalyDetectionVAE"  # noqa: E501

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: AnomalyDetectionDataset,
        from_file: bool = True,
        model_name: Optional[str] = None,
        encoder_neuron_list: List[int] = [16, 8],
        decoder_neuron_list: List[int] = [16, 32],
        latent_dim: int = 2,
        hidden_activation: str = "relu",
        output_activation: str = "sigmoid",
        batch_norm: bool = False,
        dropout_rate: float = 0.0,
        loss_beta: float = 1.0,
        loss_capacity: float = 0.0,
        clip_grad_max_norm: float = 2.0,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        num_epochs: Optional[int] = None,
        evaluate: bool = True,
    ):
        """
        Creates the model wrapper with VAE.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        dataset : AnomalyDetectionDataset
            The dataset to verify the inference.
        from_file : bool
            True if the model should be loaded from file.
        model_name : Optional[str]
            Name of the model used for the report.
        encoder_neuron_list : List[int]
            List of dense layer dimensions of encoder.
        decoder_neuron_list : List[int]
            List of dense layer dimensions of decoder.
        latent_dim : int
            Dimensions of latent layer.
        hidden_activation : str
            Activation of hidden layers.
        output_activation : str
            Activation of output layers.
        batch_norm : bool
            Whether batch norm should be enabled.
        dropout_rate : float
            Dropout rate - disabled if set to zero.
        loss_beta : float
            Parameter of the VAE loss function - scales KL divegence.
        loss_capacity : float
            Parameter of the VAE loss function - defines upper limit
            of KL divegence.
        clip_grad_max_norm: float
            Max norm for clipping gradients
        batch_size : Optional[int]
            Batch size for training.
        learning_rate : Optional[float]
            Learning rate for training.
        num_epochs : Optional[int]
            Number of epochs to train for.
        evaluate : bool
            True if the model should be evaluated each epoch.
        """
        super().__init__(model_path, dataset, from_file, model_name)

        self.encoder_neuron_list = encoder_neuron_list
        self.decoder_neuron_list = decoder_neuron_list
        self.latent_dim = latent_dim
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.loss_beta = loss_beta
        self.loss_capacity = loss_capacity
        self.clip_grad_max_norm = clip_grad_max_norm
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.evaluate = evaluate

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

    def create_model_structure(self):
        from kenning.modelwrappers.anomaly_detection.models.vae import (
            AnomalyDetectionVAE,
        )

        self.model = AnomalyDetectionVAE(
            encoder_neuron_list=self.encoder_neuron_list,
            decoder_neuron_list=self.decoder_neuron_list,
            latent_dim=self.latent_dim,
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation,
            batch_norm=self.batch_norm,
            dropout_rate=self.dropout_rate,
            input_shape=None,
            **self.model_params_from_context(self.dataset),
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
        X = (X - self.mean) / self.std
        X = X.reshape(
            -1,
            self.dataset.window_size * self.dataset.num_features,
        )
        return [X]

    def postprocess_outputs(self, y: List[Any]) -> List[np.ndarray]:
        anomalies = np.asarray(y).reshape(-1).astype(np.int8)
        return (anomalies,)

    def run_inference(self, X: List[Any]) -> List[Any]:
        import torch

        self.prepare_model()
        y = self.model(*[torch.tensor(x, device=self.device) for x in X])
        if not isinstance(y, (list, tuple)):
            y = [y]
        return [_y.detach().cpu().numpy() for _y in y]

    def train_model(self):
        import torch
        from pyod.models.vae import VAEModel

        from kenning.modelwrappers.anomaly_detection.models.vae import (
            calibrate_threshold,
            train_step,
        )

        if not self.batch_size:
            self.batch_size = self.dataset.batch_size

        missing_params = []
        if not self.learning_rate:
            missing_params.append("learning_rate")

        if not self.num_epochs:
            missing_params.append("num_epochs")

        if missing_params:
            raise TrainingParametersMissingError(missing_params)

        default_reparameterize = self.model.reparameterize
        default_batch_size = self.dataset.batch_size
        self.dataset.set_batch_size(self.batch_size)

        self.prepare_model()
        KLogger.info("Preparing dataset")
        X_train, y_train = [], []
        for X, y in self.dataset.iter_train():
            X_train += self.preprocess_input(X)
            y_train += y[0]
        X_test, y_test = [], []
        for X, y in self.dataset.iter_test():
            X_test += self.preprocess_input(X)
            y_test += y[0]
        X_train = np.concatenate(X_train)
        X_train = [
            torch.tensor(X_train[i], device=self.device)
            for i in range(X_train.shape[0])
        ]
        X_test = np.concatenate(X_test)
        X_test = [
            torch.tensor(X_test[i], device=self.device)
            for i in range(X_test.shape[0])
        ]
        contamination = sum(y_train) / len(y_train)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        prev_loss = 0
        for epoch in range(self.num_epochs):
            shuffle(X_train)
            self.model.train()
            self.model.reparameterize = lambda *x: VAEModel.reparameterize(
                self.model, *x
            )
            total_loss = 0
            with LoggerProgressBar() as logger_progress_bar:
                for batch_start in tqdm(
                    range(0, len(X_train), self.batch_size),
                    **logger_progress_bar.kwargs,
                ):
                    x = X_train[batch_start : batch_start + self.batch_size]
                    x = torch.stack(x)
                    loss, _ = train_step(
                        self.model,
                        optimizer,
                        x,
                        self.clip_grad_max_norm,
                        self.loss_beta,
                        self.loss_capacity,
                    )
                    total_loss += loss
                    if torch.isnan(loss):
                        KLogger.error(
                            "NaN loss, prev_loss: "
                            f"{prev_loss.detach().cpu().numpy()}"
                        )
                    prev_loss = loss
            mean_loss = total_loss.detach().cpu().numpy() / len(
                self.dataset.iter_train()
            )
            KLogger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} - mean_loss: {mean_loss}"
            )

            if self.evaluate or epoch == self.num_epochs - 1:
                self.model.reparameterize = default_reparameterize
                self.model.eval()
                # Calibrate threshold
                distances = []
                for batch_start in range(0, len(X_train), self.batch_size):
                    x = X_train[batch_start : batch_start + self.batch_size]
                    with torch.no_grad():
                        distance = self.model._forward_distances(
                            torch.stack(x)
                        )
                    distances.append(distance.cpu().numpy())

                calibrate_threshold(
                    self.model,
                    contamination,
                    distances,
                )

                # Evaluate model on test set
                acc = 0
                pred = []
                for batch_start in range(0, len(X_test), self.batch_size):
                    x = X_test[batch_start : batch_start + self.batch_size]
                    x = torch.stack(x)
                    with torch.no_grad():
                        anomaly = self.model(x).ravel()
                    pred.append(anomaly.cpu().numpy())
                    acc += np.sum(
                        y_test[batch_start : batch_start + self.batch_size]
                        == pred[-1]
                    )
                roc = roc_auc_score(y_test, np.concatenate(pred))
                f1 = f1_score(y_test, np.concatenate(pred))
                KLogger.info(
                    f"Test accuracy: {100 * acc / len(y_test)}%, "
                    f"ROCAUC: {roc}, F1: {f1}"
                )

        self.dataset.set_batch_size(default_batch_size)

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
                        window_size * num_features
                        if window_size > 0 and num_features > 0
                        else -1,
                    ),
                    "dtype": "float32",
                }
            ],
            "output": [
                {
                    "name": "distances",
                    "shape": (batch_size, 1),
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
    def register_components(
        cls,
        dataset: AnomalyDetectionDataset,
        platform: Platform,
    ) -> List[Type]:
        from kenning.automl.auto_pytorch_components.vae_trainer import (
            register_vae_trainer,
        )

        components = super().register_components(dataset, platform)
        register_vae_trainer()
        return components

    @classmethod
    def prepare_config(cls, configuration: Dict):
        kenning_conf = super().prepare_config(configuration)
        vae_trainer_conf = {
            k.rsplit(":", 1)[1]: v
            for k, v in configuration.items()
            if k.startswith("trainer")
        }
        for arg in [a.value for a in list(TrainerParams)]:
            if arg not in vae_trainer_conf:
                KLogger.warn(f"Missing {arg} for VAE trainer")
                continue
            kenning_conf["model_wrapper"]["parameters"][
                arg
            ] = vae_trainer_conf[arg]
        return kenning_conf

    @classmethod
    def fit(cls, self, X: Dict[str, Any], y: Any = None):
        """
        AutoPyTorch fit method.

        Overrides additional loss calculated by AutoPyTorch.
        """
        # Use binary cross entropy
        X["additional_losses"] = "BCEWithLogitsLoss"
        return super(type(self), self).fit(X, y)
