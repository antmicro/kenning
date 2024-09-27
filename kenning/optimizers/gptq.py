# Copyright (c) 2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for AutoGPTQ quantizer.

https://github.com/PanQiWei/AutoGPTQ
"""

from typing import Dict, List, Literal, Optional, Tuple

from kenning.core.dataset import Dataset
from kenning.core.optimizer import Optimizer
from kenning.sparsegpt.datautils import get_c4
from kenning.utils.resource_manager import PathOrURI


class GPTQOptimizer(Optimizer):
    """
    Optimizer subclass that provides an API
    for quantizing LLMs using AutoGPTQ optimizer.
    """

    inputtypes = {"safetensors-native": lambda x: x}

    outputtypes = ["safetensors-gptq"]

    arguments_structure = {
        "bits": {
            "description": "Target quantization precision",
            "default": 4,
            "enum": [2, 3, 4, 8],
        },
        "group_size": {
            "description": "Number of tensors that share the same "
            + "quantization parameters",
            "default": 128,
            "type": int,
        },
        "calibration_samples": {
            "description": "Number of samples in a calibration dataset",
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
        bits: int = 4,
        group_size: int = 128,
        calibration_samples: int = 128,
        desc_act: bool = True,
        symmetric: bool = True,
    ):
        self.bits = bits
        self.group_size = group_size
        self.calibration_samples = calibration_samples
        self.desc_act = desc_act

        self.symmetric = symmetric
        super().__init__(dataset, compiled_model_path, location)

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ):
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(input_model_path),
            trust_remote_code=True,
        )

        quantization_config = BaseQuantizeConfig(
            bits=self.bits,
            group_size=self.group_size,
            desc_act=self.desc_act,
            sym=self.symmetric,
        )

        model = AutoGPTQForCausalLM.from_pretrained(
            str(input_model_path), quantization_config
        )

        calibration_samples = get_c4(self.calibration_samples, tokenizer)

        model.quantize(calibration_samples)
        tokenizer.save_pretrained(str(self.compiled_model_path))
        model.save_quantized(str(self.compiled_model_path))

        self.save_io_specification(input_model_path)

    def get_framework_and_version(self) -> Tuple[str, str]:
        import auto_gptq

        return ("auto_gptq", auto_gptq.__version__)
