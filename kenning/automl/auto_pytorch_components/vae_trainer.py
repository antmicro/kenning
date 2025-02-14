# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing VAE-specific trained compatible with AutoPyTorch.
"""

from typing import Dict, Optional, Tuple, Union

import ConfigSpace as CS
import numpy as np
import torch
from autoPyTorch.constants import CLASSIFICATION_TASKS, STRING_TO_TASK_TYPES
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.training.trainer import add_trainer
from autoPyTorch.pipeline.components.training.trainer.base_trainer import (
    BaseTrainerComponent,
)
from autoPyTorch.utils.common import (
    HyperparameterSearchSpace,
    add_hyperparameter,
)
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
)
from pyod.models.vae import VAEModel

from kenning.modelwrappers.anomaly_detection.models.vae import (
    AnomalyDetectionVAE,
    calibrate_threshold,
)
from kenning.modelwrappers.anomaly_detection.models.vae import (
    train_step as _train_step,
)
from kenning.modelwrappers.anomaly_detection.vae import (
    PyTorchAnomalyDetectionVAE,
    TrainerParams,
)

_REGISTERED = False


class VAETrainer(BaseTrainerComponent):
    """
    AutoPyTorch trainer adjusted for VAE model.

    Ignores given critetion, to use custom loss for VAE.
    """

    def __init__(
        self,
        random_state: Optional[np.random.RandomState] = None,
        loss_beta: float = 1.0,
        loss_capacity: float = 0.0,
        clip_grad_max_norm: float = 2.0,
    ):
        super().__init__(random_state)
        self.beta = loss_beta
        self.capacity = loss_capacity
        self.clip_grad_max_norm = clip_grad_max_norm

    @staticmethod
    def get_properties(
        dataset_properties: Optional[
            Dict[str, BaseDatasetPropertiesType]
        ] = None,
    ) -> Dict[str, Union[str, bool]]:
        """
        Returns properties of a trainer.

        Parameters
        ----------
        dataset_properties : Optional[Dict[str, BaseDatasetPropertiesType]]
            Properties of used dataset, provided by AutoPyTorch.

        Returns
        -------
        Dict[str, Union[str, bool]]
            Trainer properties
        """
        return {
            "shortname": "VAETrainer",
            "name": "VAE Trainer",
        }

    @staticmethod
    def get_hyperparameter_search_space(
        dataset_properties: Optional[
            Dict[str, BaseDatasetPropertiesType]
        ] = None,
        beta: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter=TrainerParams.BETA.value,
            value_range=(0.0, 1.0),
            default_value=1.0,
        ),
        capacity: HyperparameterSearchSpace = HyperparameterSearchSpace(
            hyperparameter=TrainerParams.CAPACITY.value,
            value_range=(0.0, 1.0),
            default_value=0.0,
        ),
        clip_grad_max_norm: HyperparameterSearchSpace = HyperparameterSearchSpace(  # noqa: E501
            hyperparameter=TrainerParams.CLIP_GRAD.value,
            value_range=(0.1, 10.0),
            default_value=2.0,
        ),
    ) -> ConfigurationSpace:
        """
        Generates Configuration Space based on `arguments_structure`.

        Parameters
        ----------
        dataset_properties : Optional[Dict[str, BaseDatasetPropertiesType]]
            Properties of used dataset, provided by AutoPyTorch.
        beta : HyperparameterSearchSpace
            Hyperparameter defining beta parameter of loss function.
        capacity : HyperparameterSearchSpace
            Hyperparameter defining capacity parameter of loss function.
        clip_grad_max_norm : HyperparameterSearchSpace
            Hyperparameter defining maximum norm for clipping gradients.

        Returns
        -------
        ConfigurationSpace
            Configuration Space with model hyperparameter.
        """
        cs = ConfigurationSpace()
        if dataset_properties is not None:
            if (
                STRING_TO_TASK_TYPES[str(dataset_properties["task_type"])]
                in CLASSIFICATION_TASKS
            ):
                add_hyperparameter(cs, beta, UniformFloatHyperparameter)
                add_hyperparameter(cs, capacity, UniformFloatHyperparameter)
                add_hyperparameter(
                    cs, clip_grad_max_norm, UniformFloatHyperparameter
                )

        return cs

    @staticmethod
    def define_forbidden_clauses(
        cs: ConfigurationSpace,
        **kwargs: Dict,
    ) -> ConfigurationSpace:
        """
        Defines forbidden clauses for non-compatible pairs of hyperparametrs.

        Parameters
        ----------
        cs : ConfigurationSpace
            Configuration Space with hyperparameters.
        **kwargs : Dict
            Additional parameters.

        Returns
        -------
        ConfigurationSpace
            Updated Configuration Space with forbidden clauses.
        """
        network_back = cs.get_hyperparameter("network_backbone:__choice__")
        network_back_vae = CS.ForbiddenEqualsClause(
            network_back,
            PyTorchAnomalyDetectionVAE.get_component_name(),
        )

        trainer = cs.get_hyperparameter("trainer:__choice__")
        trainer.default_value = "VAETrainer"
        trainer_vae = CS.ForbiddenInClause(
            trainer, [v for v in trainer.choices if v != "VAETrainer"]
        )

        cs.add_forbidden_clause(
            CS.ForbiddenAndConjunction(
                network_back_vae,
                trainer_vae,
            )
        )

        return cs

    def data_preparation(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Prepares data for learning.

        Parameters
        ----------
        X : torch.Tensor
            The batch training features
        y : torch.Tensor
            The batch training labels

        Returns
        -------
        torch.Tensor
            Processed input data
        Dict[str, torch.Tensor]
            Arguments for the criterion function
        """
        return X, {"y_a": y}

    def _get_vae(self) -> AnomalyDetectionVAE:
        return self.model[1]

    def train_step(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[float, torch.Tensor]:
        """
        Allows to train 1 step of gradient descent,
        given a batch of train/labels.

        Parameters
        ----------
        data : torch.Tensor
            The input features to the network
        targets : torch.Tensor
            The ground truth to calculate loss

        Returns
        -------
        float
            The loss incurred in the prediction
        torch.Tensor
            The predictions of the network
        """
        data = data.float().to(self.device)
        targets = self.cast_targets(targets)

        data, _criterion_kwargs = self.data_preparation(data, targets)

        return _train_step(
            self._get_vae(),
            self.optimizer,
            data,
            self.clip_grad_max_norm,
            beta=self.beta,
            capacity=self.capacity,
        )

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int,
        writer,
    ) -> Tuple[Optional[float], Dict[str, float]]:
        model = self._get_vae()
        model._forward = model._forward_distances
        default_reparameterize = model.reparameterize
        model.reparameterize = lambda *x: VAEModel.reparameterize(model, *x)
        try:
            loss, metrics = super().train_epoch(
                train_loader=train_loader,
                epoch=epoch,
                writer=writer,
            )
            model.reparameterize = default_reparameterize
            if loss is None:
                return loss, metrics

            model.eval()
            # Calculate distances and contamination for calibration
            distances = []
            target_qt, anomalies = 0, 0
            for step, (data, targets) in enumerate(train_loader):
                if self.budget_tracker.is_max_time_reached():
                    break
                data = data.float().to(self.device)
                targets = self.cast_targets(targets)

                anomalies += torch.sum(targets).cpu().item()
                target_qt += np.prod(targets.shape)
                with torch.no_grad():
                    distance = model._forward_distances(data)
                distances.append(distance.cpu().numpy())

            calibrate_threshold(
                model,
                anomalies / target_qt,
                distances,
            )
            return loss, metrics
        finally:
            model._forward = model._forward_classify
            model.reparameterize = default_reparameterize

    def prepare(
        self,
        metrics,
        model,
        criterion,
        budget_tracker,
        optimizer,
        device,
        metrics_during_training,
        scheduler,
        task_type,
        labels,
        step_interval,
        **kwargs,
    ) -> None:
        super().prepare(
            metrics=metrics,
            model=model,
            criterion=criterion,
            budget_tracker=budget_tracker,
            optimizer=optimizer,
            device=device,
            metrics_during_training=metrics_during_training,
            scheduler=scheduler,
            task_type=task_type,
            labels=labels,
            step_interval=step_interval,
            **kwargs,
        )
        # Always disable metrics_during_training
        self.metrics_during_training = False


def register_vae_trainer():
    """
    Registers VAETrainer, only if it is not already registered.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    add_trainer(VAETrainer)
    _REGISTERED = True
