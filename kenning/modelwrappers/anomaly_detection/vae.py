# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains Variational Autoencoder (VAE) model wrapper.

Compatible with AnomalyDetectionDataset.
"""

from pathlib import Path
from random import shuffle
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

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


class PyTorchAnomalyDetectionVAE(PyTorchWrapper):
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

    default_dataset = AnomalyDetectionDataset
    arguments_structure = {
        "encoder_layers": {
            "description": "List of dense layer dimensions of encoder",
            "type": list,
            "items": int,
            "default": [16, 8],
        },
        "decoder_layers": {
            "description": "List of dense layer dimensions of decoder",
            "type": list,
            "items": int,
            "default": [16, 32],
        },
        "latent_dim": {
            "argparse_name": "--latent-dim",
            "description": "Dimensions of latent layer",
            "type": int,
            "default": 2,
        },
        "hidden_activation_name": {
            "argparse_name": "--hidden-activation",
            "description": "Activation of hidden layers",
            "type": str,
            "enum": AVAILABLE_ACTIVATION_NAMES,
            "default": "relu",
        },
        "output_activation_name": {
            "argparse_name": "--output-activation",
            "description": "Activation of output layers",
            "type": str,
            "enum": AVAILABLE_ACTIVATION_NAMES,
            "default": "sigmoid",
        },
        "batch_norm": {
            "argparse_name": "--batch-norm",
            "description": "Whether batch norm should be enabled",
            "type": bool,
            "default": False,
        },
        "dropout_rate": {
            "argparse_name": "--dropout-rate",
            "description": "Dropout rate - disabled if set to zero",
            "type": float,
            "default": 0.0,
        },
        "beta": {
            "argparse_name": "--loss-beta",
            "description": "Parameter of the VAE loss function - scales KL divegence",  # noqa: E501
            "type": float,
            "default": 1.0,
        },
        "capacity": {
            "argparse_name": "--loss-capacity",
            "description": "Parameter of the VAE loss function - defines upper limit of KL divegence",  # noqa: E501
            "type": float,
            "default": 0.0,
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: AnomalyDetectionDataset,
        from_file: bool = True,
        model_name: Optional[str] = None,
        encoder_layers: List[int] = [16, 8],
        decoder_layers: List[int] = [16, 32],
        latent_dim: int = 2,
        hidden_activation_name: str = "relu",
        output_activation_name: str = "sigmoid",
        batch_norm: bool = False,
        dropout_rate: float = 0.0,
        beta: float = 1.0,
        capacity: float = 0.0,
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
        encoder_layers : List[int]
            List of dense layer dimensions of encoder.
        decoder_layers : List[int]
            List of dense layer dimensions of decoder.
        latent_dim : int
            Dimensions of latent layer.
        hidden_activation_name : str
            Activation of hidden layers.
        output_activation_name : str
            Activation of output layers.
        batch_norm : bool
            Whether batch norm should be enabled.
        dropout_rate : float
            Dropout rate - disabled if set to zero.
        beta : float
            Parameter of the VAE loss function - scales KL divegence.
        capacity : float
            Parameter of the VAE loss function - defines upper limit
            of KL divegence.
        """
        super().__init__(model_path, dataset, from_file, model_name)

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.latent_dim = latent_dim
        self.hidden_activation_name = hidden_activation_name
        self.output_activation_name = output_activation_name
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.beta = beta
        self.capacity = capacity

        if dataset:
            self.mean, self.std = self.dataset.get_input_mean_std()

    def create_model_structure(self):
        from kenning.modelwrappers.anomaly_detection.models.vae import (
            AnomalyDetectionVAE,
        )

        self.model = AnomalyDetectionVAE(
            feature_size=self.dataset.num_features,
            window_size=self.dataset.window_size,
            encoder_neuron_list=self.encoder_layers,
            decoder_neuron_list=self.decoder_layers,
            latent_dim=self.latent_dim,
            hidden_activation_name=self.hidden_activation_name,
            output_activation_name=self.output_activation_name,
            batch_norm=self.batch_norm,
            dropout_rate=self.dropout_rate,
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

    def train_model(
        self,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        logdir: Path,
        eval: bool = True,
    ):
        import torch
        from pyod.models.vae import VAEModel, vae_loss

        default_reparameterize = self.model.reparameterize
        default_batch_size = self.dataset.batch_size
        self.dataset.set_batch_size(batch_size)

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

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        prev_loss = 0
        for epoch in range(epochs):
            shuffle(X_train)
            self.model.train()
            self.model.reparameterize = lambda *x: VAEModel.reparameterize(
                self.model, *x
            )
            total_loss = 0
            with LoggerProgressBar() as logger_progress_bar:
                for batch_start in tqdm(
                    range(0, len(X_train), batch_size),
                    file=logger_progress_bar,
                ):
                    x = X_train[batch_start : batch_start + batch_size]
                    x = torch.stack(x)
                    optimizer.zero_grad()
                    x_recon, z_mu, z_logvar = self.model.forward_minimal(x)
                    loss = vae_loss(
                        x[:, -self.dataset.num_features :],
                        x_recon,
                        z_mu,
                        z_logvar,
                        beta=self.beta,
                        capacity=self.capacity,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                    optimizer.step()
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
                f"Epoch {epoch + 1}/{epochs} - mean_loss: {mean_loss}"
            )

            if eval or epoch == epochs - 1:
                self.model.reparameterize = default_reparameterize
                self.model.eval()
                # Calibrate threshold
                distances = []
                for batch_start in range(0, len(X_train), batch_size):
                    x = X_train[batch_start : batch_start + batch_size]
                    with torch.no_grad():
                        distance = self.model.forward_distances(torch.stack(x))
                    distances.append(distance.cpu().numpy())

                distances = np.concatenate(distances)
                threshold = np.percentile(distances, 100 * (1 - contamination))
                KLogger.debug(f"Calibrated threshold: {threshold}")
                self.model.threshold = torch.nn.parameter.Parameter(
                    torch.tensor(threshold),
                    requires_grad=False,
                )

                # Evaluate model on test set
                acc = 0
                pred = []
                for batch_start in range(0, len(X_test), batch_size):
                    x = X_test[batch_start : batch_start + batch_size]
                    x = torch.stack(x)
                    with torch.no_grad():
                        anomaly = self.model(x).ravel()
                    pred.append(anomaly.cpu().numpy())
                    acc += np.sum(
                        y_test[batch_start : batch_start + batch_size]
                        == pred[-1]
                    )
                roc = roc_auc_score(y_test, np.concatenate(pred))
                f1 = f1_score(y_test, np.concatenate(pred))
                KLogger.info(
                    f"Test accuracy: {100 * acc / len(y_test)}%, "
                    f"ROCAUC: {roc}, F1: {f1}"
                )

        self.dataset.set_batch_size(default_batch_size)

    def save_model(self, model_path: PathOrURI, export_dict: bool = False):
        super().save_model(model_path, export_dict)

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
        return PyTorchAnomalyDetectionVAE._get_io_specification(
            self.dataset.num_features,
            self.dataset.window_size,
            self.dataset.batch_size,
        )
