# Copyright (c) 2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module implements functionality to run a chosen layer with GPTQ quantization.

Module is based on the vllm implementation:
https://github.com/vllm-project/vllm/blob/88407532e7ec2dd3313f6cb3a31d8dd1fa868178/vllm/model_executor/layers/quantization/gptq.py

It is only used for testing purposes and does not implement a forward pass.
"""

from typing import Optional

import custom_ext
import torch
from kenning_sparsity_aware_kernel.utils import (
    ExllamaState,
    GPTQConfig,
    assert_and_assign,
    set_weight_attrs,
)
from torch.nn.parameter import Parameter

from kenning.core.exceptions import NotSupportedError


class GPTQLayer(torch.nn.Module):
    """
    Single linear layer for GPTQ quantized model.
    """

    def __init__(self, quant_config: GPTQConfig):
        """
        Initialize the GPTQ layer.

        Parameters
        ----------
        quant_config : GPTQConfig
            The GPTQ quantization configuration.
        """
        super().__init__()
        self.quant_config = quant_config

    def create_weights(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ):
        """
        Create weights for the GPTQ linear layer.

        Parameters
        ----------
        input_size : int
            First dimension of the weight matrix.
        output_size : int
            Second dimension of the weight matrix.
        params_dtype : torch.dtype
            Data type of the parameters.
        """
        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size

        # We do not use exllama for now to simplify the flow
        self.exllama_state = ExllamaState.UNINITIALIZED

        scale_and_zero_size = input_size // group_size
        scale_and_zero_input_dim = None

        qweight = Parameter(
            torch.empty(
                input_size // self.quant_config.pack_factor,
                output_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 0,
                "output_dim": 1,
                "packed_dim": 0,
                "pack_factor": self.quant_config.pack_factor,
            },
        )

        g_idx = Parameter(
            torch.tensor(
                [i // self.quant_config.group_size for i in range(input_size)],
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        # Ignore warning from fused linear layers such as QKVParallelLinear.
        set_weight_attrs(g_idx, {"input_dim": 0, "ignore_warning": True})

        qzeros = Parameter(
            torch.empty(
                scale_and_zero_size,
                output_size // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qzeros,
            {
                "input_dim": scale_and_zero_input_dim,
                "output_dim": 1,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            },
        )
        scales = Parameter(
            torch.empty(
                scale_and_zero_size,
                output_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                "input_dim": scale_and_zero_input_dim,
                "output_dim": 1,
            },
        )

        self.register_parameter("qweight", qweight)
        self.register_parameter("g_idx", g_idx)
        self.register_parameter("qzeros", qzeros)
        self.register_parameter("scales", scales)

    def load_weights(
        self,
        qweight: torch.Tensor,
        g_idx: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
    ):
        """
        Loads the weights into prepared parameters.

        Parameters
        ----------
        qweight : torch.Tensor
            Quantized weights.
        g_idx : torch.Tensor
            Indices matrix.
        qzeros : torch.Tensor
            Zeros tensor.
        scales : torch.Tensor
            Scales tensor.
        """
        assert_and_assign(self.qweight, qweight)
        assert_and_assign(self.g_idx, g_idx)
        assert_and_assign(self.qzeros, qzeros)
        assert_and_assign(self.scales, scales)

    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through the GPTQ linear layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        bias : Optional[torch.Tensor]
            Bias tensor, by default None

        Raises
        ------
        NotSupportedError
            This is a stub method, which does not implement any logic.
            Hence, the error is raised on call.
        """
        raise NotSupportedError

    def unquantized(self) -> torch.Tensor:
        """
        Unquantize the qweights matrix.

        Returns
        -------
        torch.Tensor
            Unquantized weights.
        """
        return custom_ext.unquantize_weights(
            self.qweight,
            self.qzeros,
            self.scales,
            self.g_idx,
            4,
        )
