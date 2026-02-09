# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with simple MLP model used for testing PyTorchGenericClassification
class and optimizers.
"""

import torch


def get_model():
    """
    A function that returns a simple two layer
    MLP network.

    Return
    -------
        torch.nn.Sequential
        A simple sequential two layer network.
    """
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4),
        torch.nn.Softmax(),
    )
