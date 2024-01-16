# Copyright (c) 2023-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0


import torch
from transformers import AutoConfig

from kenning.sparsegpt.base import BasePruningConfig, BaseSparseGPTForCausalML


class MistralGPTQForCausalLM(BaseSparseGPTForCausalML):
    inside_layer_modules = ["model.embed_tokens", "model.norm"]
    outside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]


SPARSEGPT_MODEL_MAP = {
    "mistral": MistralGPTQForCausalLM,
}


class AutoSparseGPTForCausalML:
    """
    Module used to seamlessly prepare a supported model to be pruned.

    AutoSparseGPTForCausalML is designed to be instantiated
    using `AutoSparseGPTForCausalML.from_pretrained`
    if want to prune a pretrained model.
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoSparseGPTForCausalML is designed to be instantiated\n"
            "using `AutoSparseGPTForCausalML.from_pretrained` "
            "if want to prune a pretrained model."
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
        pruning_config: BasePruningConfig,
        torch_dtype: torch.dtype = torch.float16,
        dev: str = "cuda:0",
        verbosity: str = "DEBUG",
        **model_init_kwargs,
    ) -> BaseSparseGPTForCausalML:
        """
        Instantiate a sparse GPT model according to the model type.
        If the model is not supported, raise a TypeError.

        Parameters
        ----------
        pretrained_model_name_or_path : str
            Path to the pretrained model or its name.
        pruning_config : BasePruningConfig
            Pruning configuration.
        torch_dtype : torch.dtype
            Torch dtype of the model.
        dev : str
            Device on which the model is stored.
        verbosity : str
            Verbosity level.
        **model_init_kwargs :
            Keyword arguments passed to the model init function.

        Returns
        -------
        model: BaseSparseGPTForCausalML
            Sparse GPT model.
        """
        model_type = cls.check_and_get_model_type(
            pretrained_model_name_or_path, trust_remote_code=False
        )
        model_class = SPARSEGPT_MODEL_MAP[model_type]
        return model_class.from_pretrained(
            pretrained_model_name_or_path,
            pruning_config,
            torch_dtype=torch_dtype,
            dev=dev,
            verbosity=verbosity,
            **model_init_kwargs,
        )
