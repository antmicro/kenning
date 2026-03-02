# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with simple autoencoder model used for testing
PyTorchGenericAutoencoderClassification.
"""

import torch


def get_model():
    """
    A function that returns a simple
    autoencoder network.

    Return
    -------
        torch.nn.Sequential
        A simple sequential two layer network.
    """
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 128),
    )
