# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Base class for LLM optimizers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Literal, Optional

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.utils.resource_manager import PathOrURI


class LLMOptimizer(Optimizer, ABC):
    """
    Base class for LLM optimizers.
    """

    inputtypes = ["safetensors"]
    outputtypes = ["safetensors"]

    arguments_structure = {
        "model_framework": {
            "argparse_name": "--model-framework",
            "description": "The input type of the model, framework-wise",
            "default": "safetensors",
            "enum": inputtypes,
        },
    }

    def __init__(
        self,
        dataset: Optional[Dataset],
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        model_framework: str = "safetensors",
        model_wrapper: Optional[ModelWrapper] = None,
    ):
        self.model_framework = model_framework
        super().__init__(dataset, compiled_model_path, location, model_wrapper)

    @classmethod
    def get_framework(cls) -> str:
        return "safetensors"

    @classmethod
    def get_framework_version(cls) -> str:
        cls.silence_gptq_logging()
        import gptqmodel

        return gptqmodel.__version__

    @classmethod
    def silence_gptq_logging(cls) -> None:
        import warnings

        from logbar import LogBar

        LogBar.shared().setLevel("WARNING")
        warnings.filterwarnings("ignore", module="torchao")

    @abstractmethod
    def _get_quantization_config(self) -> Dict:
        ...
