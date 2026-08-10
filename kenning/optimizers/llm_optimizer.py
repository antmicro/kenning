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
        "batch_size": {
            "argparse_name": "--batch-size",
            "description": "The number of samples used in the batch during quantization",  # noqa: E501
            "default": 8,
            "type": int,
        },
        "seqlen": {
            "argparse_name": "--seqlen",
            "description": "The sequence length of samples in the calibration dataset (c4 by default)",  # noqa: E501
            "default": 1024,
            "type": int,
        },
        "calibration_samples": {
            "argparse_name": "--calibration-samples",
            "description": "The number of samples to be used from the calibration dataset",  # noqa: E501
            "type": int,
            "default": 256,
        },
    }

    def __init__(
        self,
        dataset: Optional[Dataset],
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        model_framework: str = "safetensors",
        batch_size: int = 8,
        seqlen: int = 1024,
        calibration_samples: int = 256,
        model_wrapper: Optional[ModelWrapper] = None,
    ):
        self.model_framework = model_framework
        self.batch_size = batch_size
        self.seqlen = seqlen
        self.calibration_samples = calibration_samples
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
