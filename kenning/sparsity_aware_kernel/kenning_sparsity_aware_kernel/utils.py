# Copyright (c) 2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module based on native vllm implementation so
that it can be used as a reference in benchmarks.

https://github.com/vllm-project/vllm/blob/88407532e7ec2dd3313f6cb3a31d8dd1fa868178/vllm/model_executor/layers/quantization/gptq.py
"""

import enum
from enum import Enum
from fractions import Fraction
from typing import Any, Dict, Optional

import torch


class GPTQConfig:
    """
    GPTQ config class to store GPTQ quantization parameters.
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
    ) -> None:
        """
        Initialize GPTQ config.

        Parameters
        ----------
        weight_bits : int
            Precision of the weights.
        group_size : int
            Group size for the quantization.
        desc_act : bool
            Whether to use descending order for the activations.
        """
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.pack_factor = Fraction(32, self.weight_bits)


class ExllamaState(Enum):
    """
    Determines the state of the Exllama, which may be used to
    speed up kernel execution.
    """

    UNUSED = enum.auto()
    UNINITIALIZED = enum.auto()
    READY = enum.auto()


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    """
    Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Parameters
    ----------
    weight : torch.Tensor
        The weight tensor.
    weight_attrs : Optional[Dict[str, Any]]
        A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is not None:
        for key, value in weight_attrs.items():
            assert not hasattr(
                weight, key
            ), f"Overwriting existing tensor attribute: {key}"
            setattr(weight, key, value)


def assert_and_assign(
    parameter: torch.nn.Parameter,
    tensor: torch.Tensor,
):
    """
    Check if the tensor has the same shape as the parameter and assign it.

    Parameters
    ----------
    parameter : torch.nn.Parameter
        Parameter to assign the tensor to.
    tensor : torch.Tensor
        Tensor to assign to the parameter.

    Raises
    ------
    ValueError
        If the shapes of the parameter and the tensor do not match.
    """
    if parameter.data.shape != tensor.shape:
        raise ValueError(
            f"Expected shape {parameter.data.shape}, got {tensor.shape}"
        )
    parameter.data = tensor
