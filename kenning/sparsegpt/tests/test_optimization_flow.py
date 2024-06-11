# Copyright (c) 2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from transformers import AutoTokenizer

from kenning.sparsegpt.auto import AutoSparseGPTForCausalML
from kenning.sparsegpt.base import BaseOptimizationConfig
from kenning.sparsegpt.datautils import get_c4


def test_optimization_flow_phi_2(empty_file_path: Path):
    """
    Test the optimization flow of the Phi-2 model.

    Parameters
    ----------
    empty_file_path : Path
        Path where the optimized model is stored
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
    tokenizer.save_pretrained(str(empty_file_path))
