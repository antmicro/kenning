# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains definition of simple artificial neural network
for supervised anomaly detection.
"""

from typing import List, Optional

import torch
from torch import nn


class AnomalyDetectionANN(nn.Module):
    """
    Module with ANN for anomaly detection problem.
    """

    def __init__(
        self,
        num_features: int,
        num_hidden: List[int],
        num_classes: int,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        layers = [nn.Flatten(start_dim=1)]
        in_features = num_features

        for h in num_hidden:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout:
                layers.append(nn.Dropout(dropout))
            in_features = h

        layers.extend(
            [
                nn.Linear(in_features, num_classes),
                nn.Flatten(start_dim=0),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def to_pure_torch(self) -> nn.Sequential:
        """
        Converts model to pure torch sequential model.

        Returns
        -------
        nn.Sequential
            Model in pure torch format.
        """
        return self.net
