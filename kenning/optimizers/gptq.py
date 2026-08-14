# Copyright (c) 2024-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for AutoGPTQ quantizer.

https://github.com/PanQiWei/AutoGPTQ
"""

from typing import Dict, List, Literal, Optional

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.optimizers.llm_optimizer import LLMOptimizer
from kenning.utils.resource_manager import PathOrURI


class GPTQOptimizer(LLMOptimizer):
    """
    Optimizer subclass that provides an API
    for quantizing LLMs using AutoGPTQ optimizer.
    """

    arguments_structure = {
        "bits": {
            "description": "Target quantization precision",
            "default": 4,
            "enum": [2, 3, 4, 8],
            "type": int,
        },
        "group_size": {
            "description": "Number of tensors that share the same "
            + "quantization parameters",
            "default": 128,
            "type": int,
        },
        "desc_act": {
            "description": "Determines whether to process the most "
            + "important tensors first",
            "default": True,
            "type": bool,
        },
        "symmetric": {
            "description": "Determines whether to use symmetric quantization",
            "default": False,
            "type": bool,
        },
    }

    def __init__(
        self,
        dataset: Optional[Dataset],
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        model_framework: str = "safetensors",
        bits: int = 4,
        group_size: int = 128,
        batch_size: int = 8,
        seqlen: int = 1024,
        calibration_samples: int = 256,
        desc_act: bool = True,
        symmetric: bool = True,
        model_wrapper: Optional[ModelWrapper] = None,
    ):
        self.bits = bits
        self.group_size = group_size
        self.desc_act = desc_act

        self.symmetric = symmetric
        super().__init__(
            dataset,
            compiled_model_path,
            location,
            "safetensors",
            batch_size,
            seqlen,
            calibration_samples,
            model_wrapper,
        )

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
        **kwargs: Dict,
    ):
        GPTQOptimizer.silence_gptq_logging()
        from gptqmodel import GPTQConfig, GPTQModel
        from transformers import AutoTokenizer

        from kenning.sparsegpt.datautils import get_c4

        tokenizer = AutoTokenizer.from_pretrained(
            str(input_model_path),
            trust_remote_code=True,
        )

        quantization_config = GPTQConfig(**self._get_quantization_config())

        model = GPTQModel.load(
            str(input_model_path),
            quantize_config=quantization_config,
        )

        calibration_samples = get_c4(
            n_samples=self.calibration_samples,
            tokenizer=tokenizer,
            seqlen=self.seqlen,
        )

        model.quantize(calibration_samples, batch_size=self.batch_size)
        tokenizer.save_pretrained(str(self.compiled_model_path))

        model.save(str(self.compiled_model_path))
        self.save_io_specification(input_model_path)

    def get_framework(self) -> str:
        return "safetensors"

    def _get_quantization_config(self) -> Dict:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "desc_act": self.desc_act,
            "sym": self.symmetric,
        }
