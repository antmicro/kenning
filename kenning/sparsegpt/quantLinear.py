# Copyright (c) 2024-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides functionality for packing the model with
quantized weights and sparsity metadata.
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from kenning.core.exceptions import NotSupportedError

INT32_BITS = 32
METADATA_ELEMENTS_PER_INT16 = 8


class PackingError(Exception):
    """
    Exception raised when the parameters for packing method are invalid.
    """

    pass


class QuantLinear(nn.Module):
    """
    Module representing a quantized linear layer that stores its weights
    along with optimization parameters.
    """

    def __init__(
        self,
        bits: int,
        group_size: int,
        infeatures: int,
        outfeatures: int,
        prunen: int = 0,
        prunem: int = 0,
        sparse: bool = False,
        bias: bool = False,
        development_mode: bool = False,
        weight_dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the QuantLinear layer and prepare the parameters
        buffers for weights.

        Parameters
        ----------
        bits : int
            Target precision of the weights.
        group_size : int
            Number of columns that are pruned and quantized at once.
        infeatures : int
            Number of input features.
        outfeatures : int
            Number of output features.
        prunen : int
            Number of weights pruned in semi-structured manner
            per `prunem` weights.
        prunem : int
            Block size in semi-structured pruning.
        sparse : bool
            Flag indicating if the weights should be compressed.
        bias : bool
            Flag indicating if the layer should have a bias.
        development_mode : bool
            Flag indicating if the layer should serialize the weights
            both in compressed and uncompressed format.
            Additionally, sparsity checks are performed in this mode.
            This flag may be used for debugging purposes.
        weight_dtype : torch.dtype
            Data type of the weights.

        Raises
        ------
        NotImplementedError
            Raised if the parameters passed to the constructor are invalid.
        """
        super().__init__()
        if infeatures < 8 or outfeatures < 8:
            raise NotSupportedError(
                "Pack method requires the model to "
                + "be at least of shape 8x8. "
                + "Make sure the parameters are set correctly."
            )
        if sparse and infeatures < 16:
            raise NotSupportedError(
                "Sparse quantization requires the model to "
                + "be at least of shape 16x8. "
                + "Make sure the parameters are set correctly."
            )

        if bits != 4:
            raise NotSupportedError("Only 4 bits are supported.")

        if prunem * prunen != 0 and prunen != 2 and prunem != 4:
            raise NotSupportedError("Only 2:4 pruning is supported. ")

        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.development_mode = development_mode
        self.maxq = 2**self.bits - 1

        self.compression_ratio = (
            (prunem // prunen) if (prunen * prunem != 0) and sparse else 1
        )

        # Weights are quantized to `self.bits` bits and then packed
        # into 32 bit integers. Additionally, if weights are compressed,
        # the compression ratio is taken into account.
        self.register_buffer(
            "qweight",
            torch.zeros(
                (
                    infeatures
                    * self.bits
                    // INT32_BITS
                    // self.compression_ratio,
                    outfeatures,
                ),
                dtype=torch.int32,
            ),
        )

        self.register_buffer(
            "qzeros",
            torch.zeros(
                (
                    math.ceil(infeatures / self.group_size),
                    outfeatures // INT32_BITS * self.bits,
                ),
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (math.ceil(infeatures / self.group_size), outfeatures),
                dtype=weight_dtype,
            ),
        )

        # If weights are compressed, then groups are `self.compression_ratio`
        # times smaller.
        self.register_buffer(
            "g_idx",
            torch.tensor(
                [
                    i // (self.group_size // self.compression_ratio)
                    for i in range(infeatures // self.compression_ratio)
                ],
                dtype=torch.int32,
            ),
        )

        if sparse:
            self.register_buffer(
                "qsparsity_metadata",
                torch.zeros(
                    (
                        self.outfeatures,
                        self.infeatures
                        // self.compression_ratio
                        // METADATA_ELEMENTS_PER_INT16,
                    ),
                    dtype=torch.int16,
                ),
            )

            if development_mode:
                self.register_buffer(
                    "qweight_uncompressed",
                    torch.zeros(
                        (
                            self.infeatures * self.bits // INT32_BITS,
                            self.outfeatures,
                        ),
                        dtype=torch.int32,
                    ),
                )

        else:
            self.qsparsity_metadata = None

        if bias:
            self.register_buffer(
                "bias", torch.zeros((outfeatures), dtype=weight_dtype)
            )
        else:
            self.bias = None

    def pack_sparsity_metadata(
        self, sparsity_metadata: torch.Tensor
    ) -> torch.Tensor:
        """
        Pack sparsity metadata into a compressed format.

        The format is compatible with CUDA warp-level sparse matrix storage:
        * https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-sparse-matrix-storage

        Parameters
        ----------
        sparsity_metadata : torch.Tensor
            Sparsity metadata to compress.

        Returns
        -------
        torch.Tensor
            Compressed sparsity metadata.
        """
        target_shape = (
            sparsity_metadata.shape[0],
            sparsity_metadata.shape[1]
            // self.compression_ratio
            // METADATA_ELEMENTS_PER_INT16,
        )
        # Finding indices of nonzero elements
        target_metadata = (sparsity_metadata == False).flatten().nonzero()  # noqa: E712
        target_metadata = target_metadata.reshape(-1, 8) % 4

        # Packing indices of 8 adjacent non zero elements
        target_metadata = sum(
            target_metadata[:, i] << (i * 2) for i in range(8)
        )

        target_metadata = target_metadata.reshape(target_shape)
        return target_metadata

    def pack(
        self,
        linear: nn.Linear,
        sparsity_metadata: Optional[torch.Tensor],
        scales: torch.Tensor,
        zeros: torch.Tensor,
        g_idx: torch.Tensor,
    ):
        """
        Pack the weights of the linear layer into a compressed format.

        Parameters
        ----------
        linear : nn.Linear
            Linear layer to pack.
        sparsity_metadata : Optional[torch.Tensor]
            Sparsity metadata to compress.
        scales : torch.Tensor
            Scales used for quantization.
        zeros : torch.Tensor
            Zero points used for quantization.
        g_idx : torch.Tensor
            Group indices used for quantization.

        Raises
        ------
        PackingError
            Raised if the validation of the quantization process fails.
        """
        W = linear.weight.data.clone()

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales

        # Scales and bias are stored in their native type
        self.scales[:] = scales.to(dtype=linear.weight.dtype)
        if linear.bias is not None:
            self.bias[:] = linear.bias.to(dtype=linear.weight.dtype)

        # Packing g_idx indices only if the weights are not compressed.
        # Otherwise, the indices are statically generated
        if self.qsparsity_metadata is None:
            self.g_idx[:] = g_idx

        # Quantizing weights
        intweight = []
        for idx in range(self.infeatures):
            intweight_row = torch.round(
                (W[:, idx] + scale_zeros[g_idx[idx]]) / self.scales[g_idx[idx]]
            ).to(torch.int)[:, None]

            if self.development_mode and sparsity_metadata is not None:
                dequantized = (
                    intweight_row[:, 0] - zeros[g_idx[idx]]
                ) * scales[g_idx[idx]]

                if dequantized[sparsity_metadata[:, idx]].sum() != 0:
                    raise PackingError(
                        "Error in quantization. "
                        + "Dequantized weights are not zero at zeroed indices."
                    )

            intweight.append(intweight_row)
        intweight = torch.cat(intweight, dim=1)

        if self.development_mode and self.qsparsity_metadata is not None:
            # Storing uncompressed weights as well as compressed ones
            # if the model is in development mode and is sparse
            intweight_uncompressed = intweight.clone().t().contiguous()
            intweight_uncompressed = intweight_uncompressed.numpy().astype(
                np.uint32
            )

            i = 0
            row = 0
            while row < self.qweight_uncompressed.shape[0]:
                for j in range(i, i + (INT32_BITS // self.bits)):
                    self.qweight_uncompressed[row] |= intweight_uncompressed[
                        j
                    ] << (self.bits * (j - i))
                i += INT32_BITS // self.bits
                row += 1

        if self.qsparsity_metadata is not None:
            # Compressing weight matrix
            intweight = intweight[sparsity_metadata == False].reshape(  # noqa: E712
                self.outfeatures, self.infeatures // self.compression_ratio
            )

        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        while row < self.qweight.shape[0]:
            for j in range(i, i + (INT32_BITS // self.bits)):
                self.qweight[row] |= intweight[j] << (self.bits * (j - i))
            i += INT32_BITS // self.bits
            row += 1

        # Zeros matrix has to be converted to ndarray, as torch
        # does not support required bitwise operations
        zeros = zeros.numpy().astype(np.uint32)

        # Checking whether zero points are within the range of 4 bits
        if self.development_mode and (
            np.any((zeros & ((1 << self.bits) - 1)) != zeros)
            or np.any(zeros > self.maxq)
        ):
            raise PackingError(
                "Zero points are not within the range of 4 bits."
            )

        # Packing zeros
        i = 0
        col = 0
        while col < self.qzeros.shape[1]:
            for j in range(i, i + (INT32_BITS // self.bits)):
                self.qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
            i += INT32_BITS // self.bits
            col += 1

        # Packing sparsity metadata
        if self.qsparsity_metadata is not None:
            sparsity_metadata = self.pack_sparsity_metadata(sparsity_metadata)
            self.qsparsity_metadata[:] = sparsity_metadata.to(
                dtype=torch.int16
            )

    def forward(self, x):
        raise NotSupportedError(
            "QuantLinear is not a valid layer for forward pass."
        )
