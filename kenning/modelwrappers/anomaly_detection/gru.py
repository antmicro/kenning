# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains GRU model wrapper.

Compatible with AnomalyDetectionDataset.

More information on the model can be found on
https://ieeexplore.ieee.org/document/11165323
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch import optim

from kenning.cli.command_template import TRAIN
from kenning.core.exceptions import KenningModelWrapperError
from kenning.datasets.anomaly_detection_dataset import AnomalyDetectionDataset
from kenning.modelwrappers.anomaly_detection.generic import (
    PyTorchAnomalyDetectionWrapper,
)
from kenning.utils.resource_manager import PathOrURI


def _find_best_threshold(
    true_values: np.ndarray,
    pred_scores: np.ndarray,
    anomaly_rate: float = 17.0,
) -> dict:
    """
    Simple threshold finder using a fixed positive anomaly rate.

    Parameters
    ----------
    true_values : np.ndarray
        Ground truth array.
    pred_scores : np.ndarray
        Reconstruction error array.
    anomaly_rate : float
        The percentage of anomalous data in the validation set.

    Returns
    -------
    dict
        Dictionary containing f1, threshold, percentile,
        and confusion matrix (cm).

    Raises
    ------
    KenningModelWrapperError
        Raised when the shape of the true array differ
        from the predicted array.
    """
    y = np.asarray(true_values).astype(int).ravel()
    s = np.asarray(pred_scores).ravel()

    if y.shape[0] != s.shape[0]:
        raise KenningModelWrapperError(
            f"Expected shape of predictions is {y.shape}, "
            f"got {s.shape} instead."
        )

    if y.sum() == 0:
        threshold = s.max() + 1.0
        return {
            "f1": 0.0,
            "thresholdeshold": float(threshold),
            "percentile": 0.0,
            "cm": metrics.confusion_matrix(y, s >= threshold),
        }
    if y.sum() == len(y):
        threshold = s.min() - 1.0
        return {
            "f1": 0.0,
            "thresholdeshold": float(threshold),
            "percentile": 100.0,
            "cm": metrics.confusion_matrix(y, s >= threshold),
        }

    percentage = float(anomaly_rate)
    percentile = 100.0 - percentage
    threshold = float(np.percentile(s, percentile))

    # evaluate
    yhat = s >= threshold
    f1 = float(metrics.f1_score(y, yhat))
    actual_percentage = 100.0 * yhat.mean()
    cm = metrics.confusion_matrix(y, yhat)

    return {
        "f1": f1,
        "threshold": threshold,
        "percentile": actual_percentage,
        "cm": cm,
    }


class PyTorchAnomalyDetectionGRU(PyTorchAnomalyDetectionWrapper):
    """
    Model wrapper for anomaly detection with a GRU-AE.
    """

    default_dataset = AnomalyDetectionDataset

    arguments_structure = {
        "hidden_units": {
            "argparse_name": "--hidden-units",
            "description": "Number of hidden units in the GRU layers",
            "type": int,
            "default": 64,
        },
        "num_layers": {
            "argparse_name": "--num-layers",
            "description": "Number of layers in the GRU components of the model",  # noqa: E501
            "type": int,
            "default": 2,
        },
        "dropout": {
            "argparse_name": "--dropout",
            "description": "Dropout rate",
            "type": float,
            "default": 0.1,
            "subcommands": [TRAIN],
        },
        "batch_size": {
            "argparse_name": "--batch-size",
            "description": "Batch size for training. If not assigned, dataset batch size will be used",  # noqa: E501
            "type": int,
            "default": 32,
            "subcommands": [TRAIN],
        },
        "num_epochs": {
            "argparse_name": "--num-epochs",
            "description": "Number of epochs to train for",
            "type": int,
            "default": 15,
            "subcommands": [TRAIN],
        },
        "clean_trainset": {
            "argparse_name": "--clean-trainset",
            "description": "Force the model to train only on samples labeled as normal",  # noqa: E501
            "type": bool,
            "default": True,
            "subcommands": [TRAIN],
        },
        "evaluate": {
            "argparse_name": "--evaluate",
            "description": "True if the model should be evaluated each epoch",
            "type": bool,
            "default": True,
            "subcommands": [TRAIN],
        },
        "learning_rate": {
            "argparse_name": "--learning-rate",
            "description": "Set the learning rate",
            "type": float,
            "default": 2.35e-3,
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

    model_class = "kenning.modelwrappers.anomaly_detection.models.gru.AnomalyDetectionGRU"  # noqa: E501

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: AnomalyDetectionDataset,
        from_file: bool = True,
        model_name: Optional[str] = None,
        hidden_units: int = 64,
        num_layers: int = 2,
        dropout: Optional[float] = None,
        learning_rate: float = 2.35e-3,
        batch_size: Optional[int] = None,
        num_epochs: Optional[int] = None,
        metric: str = "f1",
        clean_trainset: bool = True,
        evaluate: bool = True,
        logdir: Optional[Path] = None,
        export_dict: bool = False,
    ):
        super().__init__(
            model_path,
            dataset,
            num_epochs,
            clean_trainset,
            metric,
            learning_rate,
            from_file,
            model_name,
            logdir,
            export_dict,
        )

        self.threshold = None
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.evaluate = evaluate
        self.logdir = logdir

        self.learning_rate = learning_rate
        self.clean_trainset = clean_trainset
        self.dropout = dropout

        if dataset:
            self.mean, self.std = self.dataset.get_input_mean_std()

    def create_model_structure(self, **kwargs):
        from kenning.modelwrappers.anomaly_detection.models.gru import (
            AnomalyDetectionGRU,
        )

        self.model = AnomalyDetectionGRU(
            self.dataset.num_features,
            self.hidden_units,
            self.dataset.window_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            **kwargs,
        )

    def postprocess_outputs(self, y: List[Any]) -> List[np.ndarray]:
        # Get index of bigger value
        anomalies = (
            np.argmax(np.asarray(y), axis=-1).reshape(-1).astype(np.int8)
        )
        return (anomalies,)

    def run_inference(self, X: List[Any]) -> List[Any]:
        import numpy as np
        import torch

        self.prepare_model()
        assert self.threshold is not None, "threshold is None"

        inp = torch.tensor(X[0], device=self.device)
        if (
            inp.ndim == 4 and inp.shape[1] == 1
        ):  # (batch,1,seq,feat) -> (batch,seq,feat)
            inp = inp.squeeze(1)

        out = self.model(inp)
        scores = (
            self.model.anomaly_score(inp, out)[:, -1].detach().cpu().numpy()
        )

        # return distances in shape (batch, 2) as required by IO spec
        distances = np.stack([1.0 - scores, scores], axis=1)
        return [distances]

    def prepare_criterion(self) -> nn.Module:
        return torch.nn.MSELoss()

    def prepare_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return optim.Adam(model.parameters(), lr=self.learning_rate)

    def _forward_pass(
        self,
        criterion: torch.optim.Optimizer,
        input: torch.Tensor,
        labels: torch.Tensor,
        is_validation: bool = False,
    ):
        outputs = self.model(input)
        loss = criterion(outputs, input)

        if is_validation:
            scores = self.model.anomaly_score(input, outputs)[:, -1]
            return loss, scores
        return loss, None

    def _calculate_metrics(
        self, true: np.ndarray, predicted: np.ndarray
    ) -> (Dict[str, float], float):
        scores = _find_best_threshold(true, predicted)
        return {"f1": scores["f1"]}, scores["threshold"]

    def save_model(
        self, model_path: PathOrURI, export_dict: Optional[bool] = None
    ):
        import torch

        self.prepare_model()
        if export_dict is None:
            export_dict = self.export_dict

        payload = {
            "model_state_dict": self.model.state_dict(),
            "threshold": self.threshold,
            "mean": self.mean,
            "std": self.std,
        }

        if export_dict:
            torch.save(payload, str(model_path))
        else:
            payload["model_obj"] = self.model
            torch.save(payload, str(model_path))

    def load_model(self, model_path: PathOrURI):
        obj = torch.load(
            str(model_path), map_location=self.device, weights_only=False
        )
        if isinstance(obj, dict):
            self.create_model_structure()
            self.model.load_state_dict(obj["model_state_dict"])

            self.threshold = obj["threshold"]
            self.mean = obj["mean"]
            self.std = obj["std"]
        else:
            raise RuntimeError("Unrecognized model file format")
        self.model.to(self.device)
        self.model_prepared = True

    def get_io_specification_from_model(self) -> Dict[str, List[Dict]]:
        return self.get_io_specification_from_dataset(self.dataset)

    def get_io_specification_from_dataset(
        cls, dataset
    ) -> Dict[str, List[Dict]]:
        return cls._get_io_specification(
            dataset.num_features, dataset.window_size, dataset.batch_size
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
