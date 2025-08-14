# Copyright (c) 2023-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides wrapper for Mistral-instruct model.

https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
"""

from typing import Dict

from typing_extensions import override

from kenning.core.exceptions import NotSupportedError
from kenning.modelwrappers.llm.llm import LLM


class MistralInstruct(LLM):
    """
    Wrapper for Mistral Instruct model.

    https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
    """

    pretrained_model_uri = "hf://mistralai/Mistral-7B-Instruct-v0.1"

    @override
    def message_to_instruction(self, prompt_config: Dict | str):
        prompt_config = LLM._transform_prompt_config(prompt_config)

        if "system_message" in prompt_config:
            template = (
                "<s>[INST] {{system_message}}\n{{user_message}} [/INST] "
            )
        else:
            template = "<s>[INST] {{user_message}} [/INST] "

        return LLM._template_to_str(
            template=template, user_prompt_config=prompt_config
        )

    def train_model(self):
        raise NotSupportedError("This model does not support training.")
