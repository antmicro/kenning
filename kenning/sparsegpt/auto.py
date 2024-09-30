# Copyright (c) 2023-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Implements wrappers for models architecture that specify
which layers are to be optimized and how.

Additionally provides functionality for instantiating
a supported model according to the model type.
"""

import torch
from transformers import AutoConfig

from kenning.sparsegpt.base import (
    BaseOptimizationConfig,
    BaseSparseGPTForCausalML,
)


class MistralGPTQForCausalLM(BaseSparseGPTForCausalML):
    """
    Configuration of Mistral's OBC compression.
    """

    inside_layer_modules = ["model.embed_tokens", "model.norm"]
    outside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
    compressible_modules = [
        "mlp.up_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.q_proj",
        "self_attn.o_proj",
    ]


class PhiGPTQForCausalLM(BaseSparseGPTForCausalML):
    """
    Configuration of Phi-2's OBC compression.
    """

    inside_layer_modules = ["model.embed_tokens", "model.final_layernorm"]
    outside_layer_modules = [
        ["self_attn.q_proj"],
        ["self_attn.k_proj"],
        ["self_attn.v_proj"],
        ["self_attn.dense"],
        ["mlp.fc1"],
        ["mlp.fc2"],
    ]
    compressible_modules = [
        "mlp.fc1",
        "mlp.fc2" "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.dense",
    ]


SPARSEGPT_MODEL_MAP = {
    "mistral": MistralGPTQForCausalLM,
    "phi": PhiGPTQForCausalLM,
}


class AutoSparseGPTForCausalML:
    """
    Module used to seamlessly prepare a supported model to be optimized.

    AutoSparseGPTForCausalML is designed to be instantiated
    using `AutoSparseGPTForCausalML.from_pretrained`
    if want to optimize a pretrained model.
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoSparseGPTForCausalML is designed to be instantiated\n"
            "using `AutoSparseGPTForCausalML.from_pretrained` "
            "if want to optimize a pretrained model."
        )

    @classmethod
    def check_and_get_model_type(
        cls,
        pretrained_model_name_or_path: str,
        trust_remote_code: bool = False,
    ) -> BaseSparseGPTForCausalML:
        """
        Check if the model is supported and return its type
        using the model name or path.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Path to the pretrained model or its name.
        trust_remote_code : bool
            Whether to trust remote code.
            It is only used when the model is downloaded from HuggingFace.

        Returns
        -------
        BaseSparseGPTForCausalML
            Sparse GPT model type chosen according to the model type.

        Raises
        ------
        TypeError
            If the model is not supported.
        """
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=trust_remote_code
        )
        if config.model_type not in SPARSEGPT_MODEL_MAP:
            raise TypeError(
                f"{config.model_type} is not supported.\n",
                f"Currently supported model types are: "
                f"{list(SPARSEGPT_MODEL_MAP.keys())}",
            )
        return config.model_type

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        optimization_config: BaseOptimizationConfig,
        torch_dtype: torch.dtype = torch.float16,
        dev: str = "cuda:0",
        verbosity: str = "DEBUG",
        development_mode: bool = False,
        **model_init_kwargs,
    ) -> BaseSparseGPTForCausalML:
        """
        Instantiate a sparse GPT model according to the model type.
        If the model is not supported, raise a TypeError.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Path to the pretrained model or its name.
        optimization_config : BaseOptimizationConfig
            Optimization configuration.
        torch_dtype : torch.dtype
            Torch dtype of the model.
        dev : str
            Device on which the model is stored.
        verbosity : str
            Verbosity level.
        development_mode : bool
            Determines whether to run additional checks during model
            optimization. If set to True, the model will be optimized
            with additional checks to ensure that the model is optimized
            correctly.
        **model_init_kwargs :
            Keyword arguments passed to the model init function.

        Returns
        -------
        model: BaseSparseGPTForCausalML
            Sparse GPT model.
        """
        model_type = cls.check_and_get_model_type(
            str(pretrained_model_name_or_path), trust_remote_code=False
        )
        model_class = SPARSEGPT_MODEL_MAP[model_type]
        return model_class.from_pretrained(
            pretrained_model_name_or_path,
            optimization_config,
            torch_dtype=torch_dtype,
            dev=dev,
            verbosity=verbosity,
            development_mode=development_mode,
            **model_init_kwargs,
        )
