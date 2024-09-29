# Copyright (c) 2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Fixtures and functionality for sparsity kernel testing.
"""

import tempfile
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import pytest
import torch.nn as nn
from transformers import AutoTokenizer, PhiForCausalLM

from kenning.sparsegpt.auto import AutoSparseGPTForCausalML
from kenning.sparsegpt.base import BaseOptimizationConfig
from kenning.sparsegpt.datautils import get_c4


@pytest.fixture(scope="session")
def empty_file_path() -> Generator[Path, None, None]:
    """
    Fixture that returns path to a new temporary file that is closed
    automatically after using the fixture.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="session")
def quantized_compressed_phi_2_path(
    empty_file_path: Path,
) -> str:
    """
    Test the optimization flow of the Phi-2 model.

    Parameters
    ----------
    empty_file_path : Path
        Path where the optimized model is stored

    Returns
    -------
    str
        Path to the optimized model.
    """
    model_path = "microsoft/phi-2"
    config = BaseOptimizationConfig(
        sparsity=0.5,
        n_samples=128,
        prunen=2,
        prunem=4,
        bits=4,
        block_size=128,
        minlayer=None,
        maxlayer=None,
    )
    model = AutoSparseGPTForCausalML.from_pretrained(
        model_path, config, development_mode=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    data = get_c4(
        n_samples=128,
        tokenizer=tokenizer,
        seqlen=2048,
        seed_constant=5,
    )

    model.optimize(data)
    model.save_optimized(str(empty_file_path))

    return str(empty_file_path)


@pytest.fixture(scope="session")
def phi_2_layers() -> List[Tuple[Tuple[int, int], str]]:
    """
    Return the mlp layers of the Phi-2 model.

    Returns
    -------
    List[Tuple[Tuple[int, int], str]]
        List of tuples containing the shape of the layer and the layer name.
    """
    phi = PhiForCausalLM.from_pretrained("microsoft/phi-2")

    def find_layers(
        module: nn.Module,
        layers: Optional[List[nn.Module]] = None,
        name: str = "",
    ) -> Dict[str, nn.Module]:
        """
        Finds layers in a module and returns them as a dictionary.

        Parameters
        ----------
        module : nn.Module
            Module to search for layers
        layers : Optional[List[nn.Module]]
            List of layers to search for
        name : str
            Name of the layer

        Returns
        -------
        Dict[str, nn.Module]
            Dictionary with keys as layer names and values as layers
        """
        if not layers:
            layers = [nn.Linear]
        for layer in layers:
            if isinstance(module, layer):
                return {name: module}
        res = {}
        for name1, child in module.named_children():
            res.update(
                find_layers(
                    child,
                    layers=layers,
                    name=name + "." + name1 if name != "" else name1,
                )
            )
        return res

    layers = find_layers(phi)

    excluded_layers = ["lm_head", "self_attn"]

    layers_name_shape = [
        (layer.weight.shape, name)
        for name, layer in layers.items()
        if all(suffix not in name for suffix in excluded_layers)
    ]

    return layers_name_shape


def get_runners(
    model_path: str,
    layer_shape_name: Tuple[Tuple[int, int], str],
) -> Tuple:
    """
    Get dense and sparse runners for the given model and layer.

    Parameters
    ----------
    model_path : str
        Path to the model.
    layer_shape_name : Tuple[Tuple[int, int], str]
        Tuple containing the shape of the layer and the layer name.

    Returns
    -------
    Tuple
        Tuple containing the dense and sparse runners.
    """
    from kenning_sparsity_aware_kernel.gptq_runner import (
        GPTQRunner,
    )
    from kenning_sparsity_aware_kernel.utils import GPTQConfig

    layer_shape, layer_name = layer_shape_name

    qc = GPTQConfig(4, 128, False)
    dense_qr = GPTQRunner(
        qc,
        False,
        model_path,
        layer_name,
        layer_shape[1],
        layer_shape[0],
    )

    sparse_qr = GPTQRunner(
        qc,
        True,
        model_path,
        layer_name,
        layer_shape[1],
        layer_shape[0],
    )

    return dense_qr, sparse_qr
