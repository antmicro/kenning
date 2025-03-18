# Copyright (c) 2023-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides wrapper for Llama 2 model.

https://huggingface.co/meta-llama
"""

from typing import Dict, Optional, override

from kenning.core.dataset import Dataset
from kenning.modelwrappers.llm.llm import LLM
from kenning.utils.resource_manager import PathOrURI


class Llama(LLM):
    """
    Wrapper for Llama2-chat models created by Meta.

    https://huggingface.co/meta-llama
    """

    pretrained_model_uri = "hf://meta-llama/Llama-2-7B-chat-hf"

    arguments_structure = {
        "model_version": {
            "description": "Version of the model to be used",
            "type": str,
            "enum": ["7B", "13B", "70B"],
            "default": "7B",
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Optional[Dataset],
        from_file: bool = True,
        model_name: Optional[str] = None,
        model_version: str = "7B",
    ):
        """
        Initializes the Llama2 model wrapper.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        dataset : Optional[Dataset]
            The dataset to verify the inference.
        from_file : bool
            True if the model should be loaded from file.
        model_name : Optional[str]
            Name of the model used for the report
        model_version : str
            Version of the model to be used.
        """
        self.model_version = model_version
        self.pretrained_model_uri = (
            f"meta-llama/Llama-2-{self.model_version}-chat-hf"
        )
        super().__init__(model_path, dataset, from_file, model_name)

    @override
    def message_to_instruction(self, prompt_config: Dict | str) -> str:
        prompt_config = LLM._transform_prompt_config(prompt_config)

        if "system_message" in prompt_config:
            template = (
                "<s>[INST] <<SYS>>\n{{system_message}}\n<</SYS>>\n\n"
                "{{user_message}} [/INST]"
            )
        else:
            template = "<s>[INST] {user_message} [/INST] "

        return LLM._template_to_str(
            template=template, user_prompt_config=prompt_config
        )
