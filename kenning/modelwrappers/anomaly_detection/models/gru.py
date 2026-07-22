# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A GRU-based Time-Series Anomaly Detection autoencoder.

This defines a lightweight GRU Autoencoder unit for anomaly detection.
"""

from typing import Optional

import torch
from torch import Tensor, nn


class AnomalyDetectionGRU(nn.Module):
    """
    Anomaly Detection Model.
    """

    def __init__(
        self,
        d_model: int,
        hidden_units: int,
        seq_len: int,
        num_layers: int = 1,
        dropout: Optional[float] = None,
    ):
        """
        Initialize a new ``AnomalyDetectionGRU``.

        Parameters
        ----------
        d_model: int
            Number of input features in each time step of the input sequence.

        hidden_units: int
            Number of hidden units in the GRU layers.

        seq_len: int
            Length of the input sequence.

        num_layers: int
            Number of layers in the GRU components of the model.

        dropout : Optional[float]
            If None, no dropout is added to the training process. Otherwise,
            a dropout value between 0 and 1.
        """
        super().__init__()

        self.dropout_p = dropout
        if self.dropout_p:
            self.dropout = nn.Dropout(self.dropout_p)

        self.hidden_units = hidden_units
        self.seq_len = seq_len

        # Patches go to this GRU
        self.encoder = nn.GRU(
            d_model,
            hidden_units,
            num_layers=num_layers,
            batch_first=True,
        )

        # Decoder part
        self.decoder = nn.GRU(
            hidden_units, hidden_units, num_layers=num_layers, batch_first=True
        )
        self.affine = nn.Linear(hidden_units, d_model)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)

        batch, seq_len, d_model = x.shape

        # Encoder
        enc_seq, _ = self.encoder(x)
        enc_last = enc_seq[:, -1, :]

        # Decoder
        h_0 = enc_last.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        dummy_input = torch.zeros(
            batch, self.seq_len, self.hidden_units, device=x.device
        )
        dec_out, _ = self.decoder(dummy_input, h_0)

        if self.dropout_p:
            dec_out = self.dropout(dec_out)

        # Reconstruction (batch, seq_len, d_model)
        return self.affine(dec_out)

    def anomaly_score(
        self,
        labels: Tensor,
        outputs: Tensor,
    ) -> Tensor:
        score = (labels - outputs) ** 2
        return torch.mean(score, dim=2)
