"""
Module is taken from https://github.com/vllm-project/vllm/blob/a7dcc62086ea751b46b4821c2811cf8ac83711bf/vllm/model_executor/layers/quantization/gptq.py
All changes to the original code are wrapped with MODIFIED: comment.

This module is used to run compressed model with GPTQ quantization.
"""

from typing import List, Optional

# MODIFIED: Adding custom kernel import
import custom_ext
import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.linear import LinearMethodBase

# Adding GPTQConfig and ExllamaState imports
from vllm.model_executor.layers.quantization.gptq import (
    ExllamaState,
    GPTQConfig,
)
from vllm.model_executor.utils import set_weight_attrs

# MODIFIED: Constants for setting parameters shapes
# -----
COMPRESSION_RATIO = 2
METADATA_PACK_FACTOR = 8
# -----


class GPTQLinearMethod(LinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """

    def __init__(self, quant_config: GPTQConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # MODIFIED: Metadata needs to be reordered the first time it is run
        self.reordered_metadata = False

        del output_size  # Unused.
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )
        output_size_per_partition = sum(output_partition_sizes)
        if (
            output_size_per_partition % self.quant_config.pack_factor.numerator
            != 0
        ):
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size."
            )

        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size
        # MODIFIED: Changing this value to UNUSED
        exllama_state = ExllamaState.UNUSED
        scale_and_zero_size = input_size // group_size
        scale_and_zero_input_dim = None
        if (
            input_size != input_size_per_partition
            and self.quant_config.group_size != -1
        ):
            # For act-order models, we cannot use Exllama
            # for row parallel layer
            if self.quant_config.desc_act:
                exllama_state = ExllamaState.UNUSED
            else:
                # we need to partition qzeros and scales for exllama kernel
                scale_and_zero_size = input_size_per_partition // group_size
                scale_and_zero_input_dim = 0

        # MODIFIED: Size of qweight is divided by COMPRESSION_RATIO
        qweight = Parameter(
            torch.empty(
                input_size_per_partition
                // COMPRESSION_RATIO
                // self.quant_config.pack_factor,
                output_size_per_partition,
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

        # MODIFIED: Size of qweight is divided by COMPRESSION_RATIO and group
        # are COMPRESSION_RATIO times smaller
        g_idx = Parameter(
            torch.tensor(
                [
                    i // (self.quant_config.group_size // COMPRESSION_RATIO)
                    for i in range(
                        (input_size_per_partition // COMPRESSION_RATIO)
                    )
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
                output_size_per_partition // self.quant_config.pack_factor,
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
                output_size_per_partition,
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

        # MODIFIED: sparsity metadata parameter is added
        qsparsity_metadata = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition
                // COMPRESSION_RATIO
                // METADATA_PACK_FACTOR,
                dtype=torch.int16,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            qsparsity_metadata,
            {
                "input_dim": 1,
                "output_dim": 0,
            },
        )

        layer.register_parameter("qweight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("g_idx", g_idx)
        set_weight_attrs(g_idx, extra_weight_attrs)
        layer.register_parameter("qzeros", qzeros)
        set_weight_attrs(qzeros, extra_weight_attrs)
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, extra_weight_attrs)
        # MODIFIED: sparsity metadata also has extra_weight_attrs set
        layer.register_parameter("qsparsity_metadata", qsparsity_metadata)
        set_weight_attrs(qsparsity_metadata, extra_weight_attrs)

        layer.exllama_state = exllama_state

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qweight = layer.qweight
        out_shape = x.shape[:-1] + (qweight.shape[-1],)
        reshaped_x = x.reshape(-1, x.shape[-1])

        # MODIFIED: Reordeing metadata if run for the first time
        if not self.reordered_metadata:
            custom_ext.reorder_metadata(layer.qsparsity_metadata)
            self.reordered_metadata = True

        # MODIFIED: Using custom sparse kernel. Output needs
        # to be reshaped after that.

        pad_ = reshaped_x.shape[0] % 8
        if pad_ != 0 and reshaped_x.shape[0] >= 8:
            output = custom_ext.compressed_gptq_gemm(
                reshaped_x[:-pad_, :],
                layer.qweight,
                layer.qzeros,
                layer.scales,
                layer.g_idx,
                layer.qsparsity_metadata,
                self.quant_config.weight_bits,
            )
            output_pad = custom_ext.compressed_gptq_gemm(
                reshaped_x[-pad_:, :],
                layer.qweight,
                layer.qzeros,
                layer.scales,
                layer.g_idx,
                layer.qsparsity_metadata,
                self.quant_config.weight_bits,
            )
            output = torch.cat((output, output_pad), 1)
        else:
            output = custom_ext.compressed_gptq_gemm(
                reshaped_x,
                layer.qweight,
                layer.qzeros,
                layer.scales,
                layer.g_idx,
                layer.qsparsity_metadata,
                self.quant_config.weight_bits,
            )

        output = output.t().contiguous()
        # -----

        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)
