# Copyright (c) 2023-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for AutoAWQ quantizer.

https://github.com/casper-hansen/AutoAWQ
"""

from typing import Dict, List, Literal, Optional

from kenning.core.dataset import Dataset
from kenning.core.exceptions import NotSupportedError
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.utils.resource_manager import PathOrURI


def _get_awq_model(model: AutoModelForCausalLM):
    """
    Finds the appropriate AWQ class given a huggingface
    transformer model. This function is required because
    there is no straightforward way to quantize an in-memory
    safetensors model.
    """
    from awq.models.auto import AWQ_CAUSAL_LM_MODEL_MAP

    model_type = model.config.model_type
    awq_class = AWQ_CAUSAL_LM_MODEL_MAP.get(model_type)
    if awq_class is None:
        raise NotSupportedError(f"{model_type} is not supported by AWQ")

    return awq_class(
        model=model,
        model_type=model_type,
        is_quantized=False,
        config=model.config,
        quant_config={},
        processor=None,
    )


class AWQOptimizer(Optimizer):
    """
    Optimizer subclass that provides an API
    for quantizing LLMs using AutoAWQ optimizer.
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
        # AWQ supports only 4bit quantization for now,
        # which may be upgraded in the future.
        # If it is then the enum should be updated.
        "target_precision": {
            "description": "Target precision of the quantized model",
            "type": int,
            "default": 4,
            "enum": [4],
        },
        "use_zero_point": {
            "description": "Determines whether to calculate and use zero "
            + "point in quantization. If disabled, the quantized "
            + "model will be smaller, but it may affect model's accuracy.",
            "type": bool,
            "default": True,
        },
        "group_size": {
            "description": "Number of weights that share the same "
            + "quantization parameters. The higher the number, the "
            + "more memory is saved, but it may affect model's accuracy.",
            "default": 128,
            "type": int,
        },
        "mm_version": {
            "description": "Algorithm used for matrix multiplication. GEMM is "
            + "faster for large contexts, GEMV is faster for small contexts",
            "default": "GEMM",
            "enum": ["GEMM", "GEMV"],
        },
    }

    def __init__(
        self,
        dataset: Optional[Dataset],
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        model_framework: str = "safetensors",
        target_precision: int = 4,
        use_zero_point: bool = True,
        group_size: int = 128,
        mm_version: str = "GEMM",
        model_wrapper: Optional[ModelWrapper] = None,
    ):
        """
        Initialize the AWQOptimizer optimizer.

        Parameters
        ----------
        dataset : Optional[Dataset]
            Dataset used to train the model. Not used in this optimizer.
        compiled_model_path : PathOrURI
            Path or URI where compiled model will be saved.
        location : Literal["host", "target"]
            Specifies where optimization should be performed in client-server
            scenario.
        model_framework : str
            Framework of the input model, used to select a proper backend.
        target_precision : int
            Target precision of the quantized model.
        use_zero_point : bool
            Determines whether to zero point in quantization.
        group_size : int
            Number of weights that share the same quantization parameters.
        mm_version : str
            Algorithm used for matrix multiplication.
        model_wrapper : Optional[ModelWrapper]
            ModelWrapper for the optimized model (optional).
        """
        self.target_precision = target_precision
        self.use_zero_point = use_zero_point
        self.group_size = group_size
        self.mm_version = mm_version
        super().__init__(dataset, compiled_model_path, location, model_wrapper)

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
        **kwargs: Dict,
    ):
        from transformers import AutoTokenizer

        if io_spec is None:
            io_spec = self.load_io_specification(input_model_path)

        tokenizer = AutoTokenizer.from_pretrained(
            str(input_model_path),
            trust_remote_code=True,
        )

        model = converter_registry.convert(
            input_model_path, "safetensors", "safetensors"
        )

        quantization_config = {
            "w_bit": self.target_precision,
            "zero_point": self.use_zero_point,
            "q_group_size": self.group_size,
            "version": self.mm_version,
        }

        # Convert the hf safetensors model to AWQ
        model = _get_awq_model(model)

        if hasattr(self.dataset, "calib_data") and callable(
            getattr(self.dataset, "calib_data")
        ):
            model.quantize(
                tokenizer,
                quantization_config,
                calib_data=self.dataset.calib_data(),
            )
        else:
            model.quantize(tokenizer, quantization_config)

        tokenizer.save_pretrained(str(self.compiled_model_path))
        model.save_quantized(str(self.compiled_model_path))

        io_spec["quantization_algorithm"] = "AWQ"
        io_spec["quantization_config"] = model.quant_config.to_dict()
        self.save_io_specification(input_model_path, io_spec)

    def get_framework(self) -> str:
        return "safetensors"

    @classmethod
    def get_framework_version(cls) -> str:
        from awq import __version__ as awq_version

        return awq_version
