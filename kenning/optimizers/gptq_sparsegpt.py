# Copyright (c) 2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for GPTQ + sparseGPT optimizer that is compliant with
sparsity_aware_kernel.
"""

from typing import Dict, List, Literal, Optional, Tuple

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.utils.resource_manager import PathOrURI


class GPTQSparseGPTOptimizer(Optimizer):
    """
    Optimizer subclass that provides an API
    for quantizing and pruning LLMs using GPTQ + sparseGPT optimizer
    into format that is compliant with sparsity_aware_kernel.
    """

    inputtypes = {"safetensors-native": lambda x: x}

    outputtypes = ["safetensors-sparsity-aware-kernel"]

    arguments_structure = {
        "group_size": {
            "description": "Number of tensors that share the same "
            + "quantization parameters",
            "default": 128,
            "type": int,
        },
        "context_length": {
            "description": "Length of the context for the model",
            "default": 2048,
            "type": int,
        },
        "calibration_samples": {
            "description": "Number of samples in calibration dataset",
            "default": 4,
            "type": int,
        },
    }

    def __init__(
        self,
        dataset: Optional[Dataset],
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        group_size: int = 128,
        context_length: int = 2048,
        calibration_samples: int = 128,
        model_wrapper: Optional[ModelWrapper] = None,
    ):
        self.group_size = group_size
        self.context_length = context_length
        self.calibration_samples = calibration_samples

        super().__init__(dataset, compiled_model_path, location, model_wrapper)

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ):
        from transformers import AutoTokenizer

        from kenning.sparsegpt.auto import AutoSparseGPTForCausalML
        from kenning.sparsegpt.base import BaseOptimizationConfig
        from kenning.sparsegpt.datautils import get_c4

        config = BaseOptimizationConfig(
            sparsity=0.5,
            n_samples=self.calibration_samples,
            prunen=2,
            prunem=4,
            bits=4,
            block_size=self.group_size,
            minlayer=None,
            maxlayer=None,
        )
        model = AutoSparseGPTForCausalML.from_pretrained(
            str(input_model_path), config
        )
        tokenizer = AutoTokenizer.from_pretrained(str(input_model_path))

        data = get_c4(
            n_samples=self.calibration_samples,
            tokenizer=tokenizer,
            seqlen=self.context_length,
            seed_constant=5,
        )

        model.optimize(data)
        model.save_optimized(str(self.compiled_model_path))
        tokenizer.save_pretrained(str(self.compiled_model_path))

        self.save_io_specification(input_model_path)

    def get_framework_and_version(self) -> Tuple[str, str]:
        return ("kenning", "0.0.2")
