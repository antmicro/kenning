# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains implementation of Variational Autoencoder (VAE) model
for anomaly detection.
"""

import math
from typing import List, Tuple

import torch
from pyod.models.vae import LinearBlock, VAEModel
from torch import nn


class AnomalyDetectionVAE(VAEModel):
    """
    PyTorch implementation of Variational Autoencoder (VAE).
    """

    def __init__(
        self,
        feature_size: int,
        window_size: int,
        encoder_neuron_list: List[int] = [64, 16],
        decoder_neuron_list: List[int] = [16, 32],
        latent_dim: int = 2,
        hidden_activation_name: str = "relu",
        output_activation_name: str = "sigmoid",
        batch_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        """
        Creates Variational Autoencoder (VAE).

        Parameters
        ----------
        feature_size : int
            The number of features per timestamp.
        window_size : int
            The number of consecutive timestamps included in one entry.
        encoder_neuron_list : List[int]
            List of dense layer dimensions of encoder.
        decoder_neuron_list : List[int]
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
        """
        self.window_size = window_size
        super().__init__(
            feature_size,
            encoder_neuron_list,
            decoder_neuron_list,
            latent_dim,
            hidden_activation_name,
            output_activation_name,
            batch_norm,
            dropout_rate,
        )

        self.threshold = torch.nn.parameter.Parameter(
            data=torch.tensor(0.0), requires_grad=False
        )

    def _build_encoder(self) -> nn.Sequential:
        """
        Creates an encoder structure.

        Returns
        -------
        nn.Sequential
            Structure of encoder
        """
        encoder_layers = []
        last_neuron_size = self.window_size * self.feature_size
        for neuron_size in self.encoder_neuron_list:
            encoder_layers.append(
                LinearBlock(
                    last_neuron_size,
                    neuron_size,
                    activation_name=self.hidden_activation_name,
                    batch_norm=self.batch_norm,
                    dropout_rate=self.dropout_rate,
                )
            )
            last_neuron_size = neuron_size
        return nn.Sequential(*encoder_layers)

    def classify_anomaly(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Decides whether given distance indicates anomaly.

        Parameters
        ----------
        distances : torch.Tensor
            List of distances

        Returns
        -------
        torch.Tensor
            List with 1 (anomaly) and 0 (normal)
        """
        import torch

        anomaly = distances > self.threshold
        return anomaly.to(torch.int8)

    def forward_minimal(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns output of the autoencoder.

        Parameters
        ----------
        x : torch.Tensor
            Input data in shape (batch size, window size * feature size)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            * Predicted value of the last timestamp
            * Predicted mean values of distributions
            * Predicted logarithm of variances of distributions
        """
        return super().forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes input data and calculates distance between target
        and predicted value.

        Parameters
        ----------
        x : torch.Tensor
            Input data in shape (batch size, window size * feature size)

        Returns
        -------
        torch.Tensor
            Calculated distances
        """
        x_recon, _, _ = super().forward(x)
        euclidean_sq = torch.square(x_recon - x[:, -self.feature_size :])
        distance = torch.sqrt(torch.sum(euclidean_sq, axis=1)).reshape((-1, 1))
        return distance

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Generates random number from normal distribution
        defined by given mean values and logarithm of variance.

        As TFLite Micro and microTVM do not support random number generation,
        Box-Muller transform was used with pseudo-random values
        taken from two consecutive fields of given variances.

        Parameters
        ----------
        mu : torch.Tensor
            Mean values of the distribution
        logvar : torch.Tensor
            Logarithm of variances of the distribution

        Returns
        -------
        torch.Tensor
            Random number from normal distribution
        """
        std = torch.exp(0.5 * logvar)
        max = torch.max(std)
        std_norm = std / max
        eps = torch.sqrt(-2 * torch.log(std_norm[:-1])) * torch.cos(
            2 * math.pi * std_norm[1:]
        )
        eps = torch.cat(
            [
                eps,
                torch.sqrt(-2 * torch.log(std_norm[-1:]))
                * torch.cos(2 * math.pi * std_norm[0:1]),
            ]
        )
        return mu + eps.detach() * std
