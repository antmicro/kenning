# Copyright (c) 2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for running GPTQ layers.
"""

import torch
from kenning_sparsity_aware_kernel.layers.gptq_compressed_layer import (
    GPTQCompressedLayer,
)
from kenning_sparsity_aware_kernel.layers.gptq_layer import (
    GPTQLayer,
)
from kenning_sparsity_aware_kernel.utils import GPTQConfig
from safetensors import safe_open


class GPTQRunner(torch.nn.Module):
    """
    Runner of models quantized with GPTQ algorithm.
    """

    def __init__(
        self,
        quant_config: GPTQConfig,
        compressed: bool,
        model_dir: str,
        layer_to_run: str,
        layer_input: int,
        layer_output: int,
    ):
        """
        Initializes GPTQ runner.

        Parameters
        ----------
        quant_config : GPTQConfig
            Quantization configuration.
        compressed : bool
            Whether the model is both quantized and compressed.
        model_dir : str
            Path to the model directory.
        layer_to_run : str
            Name of the layer to run.
        layer_input : int
            First dimension of the layer.
        layer_output : int
            Second dimension of the layer
        """
        super().__init__()
        self.quant_config = quant_config

        # Layer being benchmarked
        model_weights_filename = f"{model_dir}/model.safetensors"

        tensors = {}
        with safe_open(
            model_weights_filename,
            framework="pt",
            device=0,
        ) as f:
            for tensor in f.keys():
                if tensor.startswith(layer_to_run):
                    tensors[tensor] = f.get_tensor(tensor)

        if compressed:
            self.quant_layer = GPTQCompressedLayer(quant_config)
        else:
            self.quant_layer = GPTQLayer(quant_config)

        # Creating weights for the layer
        self.quant_layer.create_weights(
            layer_input,
            layer_output,
            torch.float16,
        )

        # Loading weights
        if compressed:
            self.quant_layer.load_weights(
                tensors[f"{layer_to_run}.qweight"],
                tensors[f"{layer_to_run}.g_idx"],
                tensors[f"{layer_to_run}.qzeros"],
                tensors[f"{layer_to_run}.scales"],
                tensors[f"{layer_to_run}.qsparsity_metadata"],
            )
        else:
            g_idx = tensors[f"{layer_to_run}.g_idx"]

            # If model was optimized in development_mode,
            # the uncompressed weights are stored as `qweight_uncompressed`
            # and group indices matrix has to be twice as long, as groups
            # has twice as many elements
            if f"{layer_to_run}.qweight_uncompressed" in tensors:
                qweights = tensors[f"{layer_to_run}.qweight_uncompressed"]
                g_idx = g_idx.repeat(2, 1).t().reshape(-1)
            else:
                qweights = tensors[f"{layer_to_run}.qweight"]

            self.quant_layer.load_weights(
                qweights,
                g_idx,
                tensors[f"{layer_to_run}.qzeros"],
                tensors[f"{layer_to_run}.scales"],
            )

    def forward(self, x):
        """
        Forward pass through the runner.
        """
        return self.quant_layer(x)

    def unquantized(self):
        """
        Returns unqauntized and potentially unsparsified weights.
        """
        return self.quant_layer.unquantized()
