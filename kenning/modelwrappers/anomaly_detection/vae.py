# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains Variational Autoencoder (VAE) model wrapper.

Compatible with AnomalyDetectionDataset.
"""

from enum import Enum
from pathlib import Path
from random import shuffle
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

from kenning.automl.auto_pytorch import AutoPyTorchModel
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

    default_dataset = AnomalyDetectionDataset
    arguments_structure = {
        "encoder_neuron_list": {
            "description": "List of dense layer dimensions of encoder",
            "type": list,
            "items": int,
            "default": [16, 8],
            "AutoML": True,
            "list_range": (2, 6),
            "item_range": (4, 48),
        },
        "decoder_neuron_list": {
            "description": "List of dense layer dimensions of decoder",
            "type": list,
            "items": int,
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
        },
        TrainerParams.CAPACITY.value: {
            "description": "Parameter of the VAE loss function - defines upper limit of KL divegence",  # noqa: E501
            "type": float,
            "default": 0.0,
        },
        TrainerParams.CLIP_GRAD.value: {
            "description": "Max norm for clipping gradients",
            "type": float,
            "default": 2.0,
        },
    }

    model_class = "AnomalyDetectionVAE"

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

        if dataset:
            self.mean, self.std = self.dataset.get_input_mean_std()

    @staticmethod
    def _create_model_structure(
        input_shape: Tuple[int, ...],
        encoder_neuron_list: List[int],
        decoder_neuron_list: List[int],
        latent_dim: int,
        hidden_activation: str,
        output_activation: str,
        batch_norm: bool,
        dropout_rate: float,
        dataset: AnomalyDetectionDataset,
    ):
        from kenning.modelwrappers.anomaly_detection.models.vae import (
            AnomalyDetectionVAE,
        )

        return AnomalyDetectionVAE(
            window_size=dataset.window_size,
            feature_size=dataset.num_features,
            encoder_neuron_list=encoder_neuron_list,
            decoder_neuron_list=decoder_neuron_list,
            latent_dim=latent_dim,
            hidden_activation_name=hidden_activation,
            output_activation_name=output_activation,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
        )

    def create_model_structure(self):
        self.model = self._create_model_structure(
            input_shape=None,
            encoder_neuron_list=self.encoder_neuron_list,
            decoder_neuron_list=self.decoder_neuron_list,
            latent_dim=self.latent_dim,
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation,
            batch_norm=self.batch_norm,
            dropout_rate=self.dropout_rate,
            dataset=self.dataset,
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
        from pyod.models.vae import VAEModel

        from kenning.modelwrappers.anomaly_detection.models.vae import (
            calibrate_threshold,
            train_step,
        )

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

    @classmethod
    def register_components(
        cls, dataset: Optional[AnomalyDetectionDataset] = None
    ) -> List[Type]:
        from kenning.automl.auto_pytorch_components.network_head_passthrough import (  # noqa: E501
            register_passthrough,
        )
        from kenning.automl.auto_pytorch_components.vae_trainer import (
            register_vae_trainer,
        )

        components = super().register_components(dataset)
        register_passthrough()
        register_vae_trainer()
        return components

    @staticmethod
    def define_forbidden_clauses(cs, **kwargs):
        import ConfigSpace as CS

        from kenning.automl.auto_pytorch_components.utils import (
            _create_forbidden_choices,
        )

        vae_component = PyTorchAnomalyDetectionVAE.get_component_name()
        network_back = cs.get_hyperparameter("network_backbone:__choice__")
        vae_only = (
            len(network_back.choices) == 1
            and vae_component in network_back.choices
        )
        network_back_vae = CS.ForbiddenEqualsClause(
            network_back,
            vae_component,
        )

        clauses = [
            _create_forbidden_choices(cs, name, (choice,), vae_only)
            for name, choice in (
                ("imputer:numerical_strategy", "constant_zero"),
                ("network_head:__choice__", "PassthroughHead"),
                ("network_embedding:__choice__", "NoEmbedding"),
                ("feature_preprocessor:__choice__", "NoFeaturePreprocessor"),
                ("encoder:__choice__", "NoEncoder"),
                ("coalescer:__choice__", "NoCoalescer"),
                ("scaler:__choice__", "StandardScaler"),
            )
        ]

        cs.add_forbidden_clauses(
            [
                CS.ForbiddenAndConjunction(
                    network_back_vae,
                    clause,
                )
                for clause in clauses
            ]
        )
        return cs

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
