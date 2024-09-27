# Copyright (c) 2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for testing sparse metadata operations.
"""

from typing import List, Tuple

import pytest


@pytest.mark.gpu
def test_metadata(
    phi_2_layers: List[Tuple[Tuple[int, int], str]],
    quantized_compressed_phi_2_path: str,
):
    """
    Test the metadata of the Phi-2 model
    by comparing the sparse and dense weights.

    Parameters
    ----------
    phi_2_layers : List[Tuple[Tuple[int, int], str]]
        List of tuples containing the shape of the layer and the layer name.
    quantized_compressed_phi_2_path : str
        Path to the compressed and quantized Phi-2 model.
    """
    import torch
    from kenning_sparsity_aware_kernel.tests.conftest import (
        get_runners,
    )

    for layer_shape_name in phi_2_layers:
        qr_dense, qr_sparse = get_runners(
            quantized_compressed_phi_2_path, layer_shape_name
        )

        sparse_weight = qr_sparse.unquantized()
        dense_weight = qr_dense.unquantized()

        # Sparse weight matrix has to be reshaped as its column-wise
        assert torch.equal(
            sparse_weight.reshape(dense_weight.shape), dense_weight
        )
