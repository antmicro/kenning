# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Contains definition of simple Convolution Neural Network (CNN) architecture
for anomaly detection problem.
"""

from pathlib import Path
from typing import List, Literal, Optional, Tuple

import torch
from torch import nn

from kenning.utils.class_loader import append_to_sys_path

# Mapping of ai8x-training names to Torch pooling layers
TORCH_POOLING = {
    "Max": nn.MaxPool2d,
    "Avg": nn.AvgPool2d,
}


class Abs(nn.Module):
    """
    Module representing Abs activation.
    """

    def forward(self, x):
        return torch.abs(x)


# Mapping of ai8x-training names to Torch activation layers
TORCH_ACTIVATION = {
    "ReLU": nn.ReLU,
    "Abs": Abs,
}


class AnomalyDetectionCNN(nn.Sequential):
    """
    Module with CNN for anomaly detection problem.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        feature_size: int,
        window_size: int,
        filters: List[int] = [8, 16],
        kernel_size: int = 3,
        conv_padding: int = 0,
        conv_stride: int = 1,
        conv_dilation: int = 1,
        conv_activation: Optional[Literal["ReLU", "Abs"]] = None,
        conv_batch_norm: Optional[Literal["Affine", "NoAffine"]] = None,
        pooling: Optional[Literal["Max", "Avg"]] = None,
        pool_size: int = 2,
        pool_stride: int = 2,
        pool_dilation: int = 1,
        fc_neurons: List[int] = [8],
        fc_activation: Optional[Literal["ReLU", "Abs"]] = None,
        quantize_activation: bool = False,
        qat: bool = False,
        qat_weight_bits: Optional[int] = None,
        ai8x_training_path: Optional[Path] = None,
    ):
        self.feature_size = feature_size
        self.window_size = window_size

        self.filters = filters
        self.kernel_size = kernel_size
        self.conv_padding = conv_padding
        self.conv_stride = conv_stride
        self.conv_dilation = conv_dilation
        self.conv_batch_norm = conv_batch_norm
        self.conv_activation = conv_activation

        self.pooling = pooling
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.pool_dilation = pool_dilation

        self.fc_neurons = fc_neurons
        self.fc_activation = fc_activation

        self.quantize_activation = quantize_activation
        self.qat = qat
        self.qat_weight_bits = qat_weight_bits
        self.ai8x_training_path = ai8x_training_path

        self._conv_layers = self._prepare_conv()
        super().__init__(*self._conv_layers, nn.Flatten(), *self._prepare_fc())

    def _prepare_conv(self) -> List[nn.Module]:
        channels = [1] + self.filters
        if self.qat:
            with append_to_sys_path([self.ai8x_training_path]):
                import ai8x

            return [
                ai8x.Conv2d(
                    in_channels=channel,
                    out_channels=channels[i + 1],
                    kernel_size=self.kernel_size,
                    stride=self.conv_stride,
                    padding=self.conv_padding,
                    dilation=self.conv_dilation,
                    activation=self.conv_activation,
                    weight_bits=self.qat_weight_bits,
                    bias_bits=self.qat_weight_bits,
                    quantize_activation=self.quantize_activation,
                    batchnorm=self.conv_batch_norm,
                    **(
                        {  # Add pooling for last Conv
                            "pooling": self.pooling,
                            "pool_size": self.pool_size,
                            "pool_stride": self.pool_stride,
                            "pool_dilation": self.pool_dilation,
                        }
                        if i + 1 == len(self.filters)
                        else {}
                    ),
                )
                for i, channel in enumerate(channels[:-1])
            ]

        layers = []
        for i, channel in enumerate(channels[:-1]):
            layers.append(
                nn.Conv2d(
                    in_channels=channel,
                    out_channels=channels[i + 1],
                    kernel_size=self.kernel_size,
                    stride=self.conv_stride,
                    padding=self.conv_padding,
                    dilation=self.conv_dilation,
                )
            )
            if self.conv_batch_norm:
                layers.append(
                    nn.BatchNorm2d(
                        num_features=channels[i + 1],
                        affine=self.conv_batch_norm == "Affine",
                    )
                )
            if self.conv_activation:
                layers.append(TORCH_ACTIVATION[self.conv_activation]())
        if self.pooling:
            layers.append(
                TORCH_POOLING[self.pooling](
                    kernel_size=self.pool_size,
                    stride=self.pool_stride,
                    **(
                        {"dilation": self.pool_dilation}
                        if self.pooling != "Avg"
                        else {}
                    ),
                )
            )
        return layers

    def _prepare_fc(self) -> List[nn.Module]:
        # Pass random input through Conv to get size for FC layers
        rand_input = torch.rand((2, 1, self.window_size, self.feature_size))
        rand_output = nn.Sequential(*self._conv_layers)(rand_input)
        rand_output = torch.flatten(rand_output, start_dim=1)

        neurons = [rand_output.shape[1]] + self.fc_neurons + [2]
        if self.qat:
            with append_to_sys_path([self.ai8x_training_path]):
                import ai8x

            return [
                ai8x.Linear(
                    in_features=neuron,
                    out_features=neurons[i + 1],
                    activation=self.fc_activation,
                )
                for i, neuron in enumerate(neurons[:-1])
            ]

        layers = []
        for i, neuron in enumerate(neurons[:-1]):
            layers.append(nn.Linear(neuron, neurons[i + 1]))
            if self.fc_activation:
                layers.append(TORCH_ACTIVATION[self.fc_activation]())
        return layers
