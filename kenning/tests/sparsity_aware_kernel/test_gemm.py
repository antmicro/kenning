# Copyright (c) 2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for testing sparse gemm operations.
"""

import logging
from typing import List, Tuple

import pytest

NUM_OF_RUNS = 25
logger = logging.getLogger(__name__)


@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 6, 7, 8, 13, 1024])
def test_gemm(
    phi_2_layers: List[Tuple[Tuple[int, int], str]],
    quantized_compressed_phi_2_path: str,
    batch_size: int,
):
    """
    Test the GEMM operation of the Phi-2 model.

    Parameters
    ----------
    phi_2_layers : List[Tuple[Tuple[int, int], str]]
        List of tuples containing the shape of the layer and the layer name.
    quantized_compressed_phi_2_path : str
        Path to the compressed and quantized Phi-2 model.
    batch_size : int
        Batch size for the input.
    """
    import torch
    from kenning_sparsity_aware_kernel.tests.conftest import (
        get_runners,
    )

    for layer_shape_name in phi_2_layers:
        layer_shape, layer_name = layer_shape_name
        x = (
            torch.rand(
                batch_size, layer_shape[1], device="cuda", dtype=torch.float16
            )
            - 1
        ) / 100

        qr_dense, qr_sparse = get_runners(
            quantized_compressed_phi_2_path, layer_shape_name
        )

        logger.info(f"Input layer of size {x.shape[0]}x{x.shape[1]}")
        logger.info(
            f"Weight layer {layer_name} of size "
            + f"{layer_shape[1]}x{layer_shape[0]}"
        )

        y_sparse = qr_sparse(x)

        # Calculating the result using pytorch implementation
        y_pytorch = x @ qr_dense.unquantized()

        assert torch.allclose(
            y_pytorch,
            y_sparse.t(),
            rtol=0.01,
            atol=0.001,
        )
