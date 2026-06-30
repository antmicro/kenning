# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
Contains a wrapper model for a simple supervised anomaly
detection model.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics

from kenning.cli.command_template import TRAIN
from kenning.datasets.anomaly_detection_dataset import AnomalyDetectionDataset
from kenning.modelwrappers.anomaly_detection.generic import (
    PyTorchAnomalyDetectionWrapper,
)
from kenning.utils.resource_manager import PathOrURI


class PyTorchAnomalyDetectionANN(PyTorchAnomalyDetectionWrapper):
    """
    Model wrapper for anomaly detection with an artificial
    neural network.
    """

    default_dataset = AnomalyDetectionDataset

    model_class = "kenning.modelwrappers.anomaly_detection.models.ann.AnomalyDetectionANN"  # noqa: E501

    arguments_structure = {
        "threshold": {
            "argparse_name": "--threshold",
            "description": "Binary decision threshold",
            "type": float,
            "default": 0.5,
        },
        "hidden_layers": {
            "argparse_name": "--hidden-layers",
            "description": "Dimensions of the hidden layers",
            "type": List[int],
            "default": [],
        },
        "dropout": {
            "argparse_name": "--dropout",
            "description": "Dropout rate",
            "type": float,
            "default": None,
            "subcommands": [TRAIN],
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: AnomalyDetectionDataset,
        hidden_layers: List[int] = [],
        threshold: float = 0.5,
        from_file: bool = True,
        dropout: Optional[float] = None,
        model_name: Optional[str] = None,
        metric: str = "f1",
        learning_rate: float = 1e-3,
        batch_size: Optional[int] = None,
        num_epochs: Optional[int] = None,
        evaluate: bool = True,
        logdir: Optional[Path] = None,
        export_dict: bool = False,
        **kwargs,
    ):
        super().__init__(
            model_path,
            dataset,
            num_epochs,
            False,  # trainset should not be clean.
            metric,
            learning_rate,
            from_file,
            model_name,
            logdir,
            export_dict,
        )

        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.threshold = threshold

        self.batch_size = batch_size
        self.evaluate = evaluate

    def create_model_structure(self, **kwargs):
        from kenning.modelwrappers.anomaly_detection.models.ann import (
            AnomalyDetectionANN,
        )

        num_features = self.dataset.num_features
        window_size = self.dataset.window_size
        self.model = AnomalyDetectionANN(
            num_features * window_size,  # Input will be flatteened.
            self.hidden_layers,
            1,  # Binary Classification
            self.dropout,
        )

    def prepare_criterion(self) -> nn.Module:
        return nn.BCEWithLogitsLoss()

    def prepare_optimizer(self, model: nn.Module) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=self.learning_rate)

    def run_inference(self, X: List[Any]) -> List[Any]:
        self.prepare_model()
        inp = torch.tensor(X[0], device=self.device)
        inp = inp.squeeze(1)
        out = self.model(inp)
        return [out.detach().cpu().numpy()]
        # return [(out >= self.threshold).to(inp.dtype)]

    def postprocess_outputs(self, y: List[Any]):
        assert y != [], "y is empty"
        return ((y[0] >= self.threshold).astype(np.int8),)

    def _forward_pass(
        self,
        criterion: nn.Module,
        inputs: Any,
        labels: Any,
        is_validation: bool = False,
    ) -> (torch.Tensor, Any):
        outputs = self.model(inputs)
        loss = criterion(outputs, labels.to(outputs.dtype))
        if is_validation:
            probs = F.sigmoid(outputs)
            return loss, probs
        return loss, None

    def _calculate_metrics(
        self, true: np.ndarray, predicted: np.ndarray
    ) -> (Dict[str, float], float):
        predicted = (predicted >= self.threshold).astype(int)
        return {"f1": metrics.f1_score(true, predicted)}, self.threshold

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
                    "name": "logits",
                    "shape": (batch_size,),
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
