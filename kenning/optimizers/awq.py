# Copyright (c) 2023-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for AutoAWQ quantizer.

https://github.com/casper-hansen/AutoAWQ
"""

from typing import Dict, List, Literal, Optional

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.optimizers.llm_optimizer import LLMOptimizer
from kenning.utils.resource_manager import PathOrURI


class AWQOptimizer(LLMOptimizer):
    """
    Optimizer subclass that provides an API
    for quantizing LLMs using AutoAWQ optimizer.
    """

    arguments_structure = {
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
        batch_size: int = 8,
        seqlen: int = 1024,
        calibration_samples: int = 256,
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
        batch_size : int
            The number of samples used in the batch during quantization.
        seqlen : int
            The sequence length of samples in the calibration dataset
            (c4 by default).
        calibration_samples : int
            The number of samples to be used from the calibration dataset.
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
        super().__init__(
            dataset,
            compiled_model_path,
            location,
            model_framework,
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
        AWQOptimizer.silence_gptq_logging()
        from gptqmodel import AWQConfig, GPTQModel
        from transformers import AutoTokenizer

        if io_spec is None:
            io_spec = self.load_io_specification(input_model_path)

        tokenizer = AutoTokenizer.from_pretrained(
            str(input_model_path),
            trust_remote_code=True,
        )

        quantization_config = AWQConfig(**self._get_quantization_config())

        model = GPTQModel.load(
            str(input_model_path),
            quantize_config=quantization_config,
            device_map="auto",
        )

        calib_data = None
        if hasattr(self.dataset, "calib_data") and callable(
            getattr(self.dataset, "calib_data")
        ):
            calib_data = self.dataset.calib_data()

        if calib_data is None:
            from kenning.sparsegpt.datautils import get_c4

            calib_data = get_c4(
                n_samples=self.calibration_samples,
                tokenizer=tokenizer,
                seqlen=self.seqlen,
            )

        model.quantize(calib_data, batch_size=self.batch_size)

        tokenizer.save_pretrained(str(self.compiled_model_path))
        model.save(str(self.compiled_model_path))

        io_spec["quantization_algorithm"] = "AWQ"
        io_spec["quantization_config"] = quantization_config.to_dict()
        io_spec["quantization_config"]["version"] = self.mm_version

        self.save_io_specification(input_model_path, io_spec)

    def get_framework(self) -> str:
        return "safetensors"

    def _get_quantization_config(self) -> Dict:
        AWQOptimizer.silence_gptq_logging()
        from gptqmodel.quantization.config import FORMAT

        return {
            "bits": self.target_precision,
            "group_size": self.group_size,
            "sym": not self.use_zero_point,
            "format": getattr(FORMAT, self.mm_version, FORMAT.GEMM),
        }
