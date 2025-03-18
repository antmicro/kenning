# Copyright (c) 2023-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides wrapper for Phi-2 model.

https://huggingface.co/microsoft/phi-2
"""

from typing import Dict, override

from kenning.modelwrappers.llm.llm import LLM


class PHI2(LLM):
    """
    Wrapper for Phi-2 model.

    https://huggingface.co/microsoft/phi-2
    """

    pretrained_model_uri = "hf://microsoft/phi-2"

    @override
    def message_to_instruction(self, prompt_config: Dict | str) -> str:
        prompt_config = LLM._transform_prompt_config(prompt_config)

        if "system_message" in prompt_config:
            template = (
                "Instruct: {{system_message}}. {{user_message}}\nOutput:"
            )
        else:
            template = "Instruct: {{user_message}}\nOutput:"

        return LLM._template_to_str(
            template=template, user_prompt_config=prompt_config
        )
