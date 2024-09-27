# Copyright (c) 2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module implements functionality to run a chosen
layer with GPTQ quantization and pruning.

Module is based on the vllm implementation:
https://github.com/vllm-project/vllm/blob/88407532e7ec2dd3313f6cb3a31d8dd1fa868178/vllm/model_executor/layers/quantization/gptq.py
"""

from typing import Optional

import custom_ext
import torch
from kenning_sparsity_aware_kernel.utils import (
    GPTQConfig,
    assert_and_assign,
    set_weight_attrs,
)
from torch.nn.parameter import Parameter

COMPRESSION_RATIO = 2
METADATA_PACK_FACTOR = 8


class GPTQCompressedLayer(torch.nn.Module):
    """
    Single linear layer for GPTQ quantized and compressed model.
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

        self.reordered_metadata = False

        scale_and_zero_size = input_size // group_size
        scale_and_zero_input_dim = None

        qweight = Parameter(
            torch.empty(
                input_size
                // COMPRESSION_RATIO
                // self.quant_config.pack_factor,
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

        # NOTE: g_idx is two times smaller than in the uncompressed case
        g_idx = Parameter(
            torch.tensor(
                [
                    (i // (self.quant_config.group_size // COMPRESSION_RATIO))
                    for i in range(input_size // COMPRESSION_RATIO)
                ],
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

        qsparsity_metadata = Parameter(
            torch.empty(
                output_size,
                input_size // COMPRESSION_RATIO // METADATA_PACK_FACTOR,
                dtype=torch.int16,
            ),
            requires_grad=False,
        )

        self.register_parameter("qweight", qweight)
        self.register_parameter("g_idx", g_idx)
        self.register_parameter("qzeros", qzeros)
        self.register_parameter("scales", scales)
        self.register_parameter("qsparsity_metadata", qsparsity_metadata)

    def load_weights(
        self,
        qweight: torch.Tensor,
        g_idx: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        qsparsity_metadata: torch.Tensor,
    ):
        """
        Loads the weights into prepared parameters.

        Parameters
        ----------
        qweight : torch.Tensor
            Quantized and compressed weights.
        g_idx : torch.Tensor
            Indices matrix.
        qzeros : torch.Tensor
            Zeros tensor.
        scales : torch.Tensor
            Scales tensor.
        qsparsity_metadata : torch.Tensor
            Sparsity metadata.
        """
        assert_and_assign(self.qweight, qweight)
        assert_and_assign(self.g_idx, g_idx)
        assert_and_assign(self.qzeros, qzeros)
        assert_and_assign(self.scales, scales)
        assert_and_assign(self.qsparsity_metadata, qsparsity_metadata)

    def prepare_metadata(self):
        """
        Reorders metadata and prepares it for the inference.

        This function has to be run before the first forward pass.
        It will be call automatically by the forward function if not
        done explicitly.
        """
        if not self.reordered_metadata:
            custom_ext.reorder_metadata(self.qsparsity_metadata)
            self.reordered_metadata = True

    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the compressed GPTQ linear layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        bias : Optional[torch.Tensor]
            Bias tensor, by default None

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        qweight = self.qweight
        out_shape = (qweight.shape[1],) + (x.shape[0],)
        reshaped_x = x.reshape(-1, x.shape[-1])

        # When running for the first time, reorder the metadata
        if not self.reordered_metadata:
            self.prepare_metadata()

        pad_ = reshaped_x.shape[0] % 8
        if pad_ != 0 and reshaped_x.shape[0] >= 8:
            output = custom_ext.compressed_gptq_gemm(
                reshaped_x[:-pad_, :],
                self.qweight,
                self.qzeros,
                self.scales,
                self.g_idx,
                self.qsparsity_metadata,
                self.quant_config.weight_bits,
            )
            output_pad = custom_ext.compressed_gptq_gemm(
                reshaped_x[-pad_:, :],
                self.qweight,
                self.qzeros,
                self.scales,
                self.g_idx,
                self.qsparsity_metadata,
                self.quant_config.weight_bits,
            )
            output = torch.cat((output, output_pad), 1)
        else:
            output = custom_ext.compressed_gptq_gemm(
                reshaped_x,
                self.qweight,
                self.qzeros,
                self.scales,
                self.g_idx,
                self.qsparsity_metadata,
                self.quant_config.weight_bits,
            )

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)

    def unquantized(self, uncompress: bool = True) -> torch.Tensor:
        """
        Unquantize and uncompress if enabled the qweights matrix.

        Parameters
        ----------
        uncompress : bool
            If True, uncompress the weights after unqantizing them

        Returns
        -------
        torch.Tensor
            Unquantized weights.
        """
        unquantized_compressed = custom_ext.unquantize_weights(
            self.qweight,
            self.qzeros,
            self.scales,
            self.g_idx,
            4,
        )

        if uncompress:
            return custom_ext.uncompress_weights(
                unquantized_compressed,
                self.qsparsity_metadata,
            )
        return unquantized_compressed
