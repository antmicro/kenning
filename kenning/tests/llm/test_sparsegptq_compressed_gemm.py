# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Test code for sparsegpt vllm kernels in CUDA.
"""

import torch  # noqa: I001
import custom_ext  # noqa: F401


COMPRESSION_RATIO = 2
METADATA_PACK_FACTOR = 8
BIT = 4
PACK_FACTOR = 32 // BIT


class TestQCompressedGemmSparseGPTQ:
    """
    Basic tests for the SparseGPTQ custom CUDA kernel. These tests
    ensure sanity of the kernel, and that can run on CUDA devices.
    """

    def setup_method(self, method=None):
        """
        Initialize tensor dimensions and dummy data for testing.
        """
        self.m, self.n, self.k_in, self.groups = 128, 32, 256, 4
        self.k_c = self.k_in // COMPRESSION_RATIO

        # Quantized model weights
        self.b_q_weight = torch.randint(
            0,
            100,
            (self.k_c // PACK_FACTOR, self.m),
            dtype=torch.int32,
            device="cuda",
        )
        self.b_gptq_qzeros = torch.randint(
            0,
            100,
            (self.groups, self.m // PACK_FACTOR),
            dtype=torch.int32,
            device="cuda",
        )
        self.b_gptq_scales = torch.randn(
            (self.groups, self.m), dtype=torch.float16, device="cuda"
        )
        # Group index mapping for quantization.
        self.b_g_idx = torch.zeros(
            (self.k_c,), dtype=torch.int32, device="cuda"
        )
        # Metadata defining the sparse matrix structure.
        self.sparsity_metadata = torch.randint(
            0,
            100,
            (self.m, self.k_c // METADATA_PACK_FACTOR),
            dtype=torch.int16,
            device="cuda",
        )
        # Dummy dequantized weights and input activation
        self.dq_weight = torch.randn(
            (self.k_c, self.m), dtype=torch.float16, device="cuda"
        )
        self.a = torch.randn(
            (self.n, self.k_in), dtype=torch.float16, device="cuda"
        )
        # Temporar memory buffers.
        self.workspace = torch.zeros(
            (4 * 1024 * 1024,), dtype=torch.uint8, device="cuda"
        )
        self.temp_dq = torch.empty(
            (self.k_c, self.m), dtype=torch.float16, device="cuda"
        )

    def test_unquantize_weights(self):
        """
        Test the conversion of quantized weights back to standard format.
        """
        dq_weight = torch.ops.custom_ext.unquantize_weights(
            self.b_q_weight,
            self.b_gptq_qzeros,
            self.b_gptq_scales,
            self.b_g_idx,
            BIT,
        )
        assert dq_weight.shape == (self.k_c, self.m)

    def test_reorder_metadata(self):
        """
        Test rearranging the sparsity metadata for the kernel.
        """
        torch.ops.custom_ext.reorder_metadata(self.sparsity_metadata)

    def test_uncompress_weights(self):
        """
        Test expanding the compressed weights using metadata.
        """
        uncompressed = torch.ops.custom_ext.uncompress_weights(
            self.dq_weight, self.sparsity_metadata
        )
        assert uncompressed.shape == (self.m, self.k_c * COMPRESSION_RATIO)

    def test_compressed_gptq_gemm(self):
        """
        Test the full sparse matrix multiplication pipeline.
        """
        gemm_output = torch.ops.custom_ext.compressed_gptq_gemm(
            self.a,
            self.b_q_weight,
            self.b_gptq_qzeros,
            self.b_gptq_scales,
            self.b_g_idx,
            self.sparsity_metadata,
            BIT,
            self.workspace,
            self.temp_dq,
        )
        assert gemm_output.shape == (self.m, self.n)
