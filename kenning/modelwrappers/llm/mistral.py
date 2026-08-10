# Copyright (c) 2023-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides wrapper for Mistral-instruct model.

https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
"""


from kenning.modelwrappers.llm.llm import LLM


class MistralInstruct(LLM):
    """
    Wrapper for Mistral Instruct model.

    https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
    """

    pretrained_model_uri = "hf://ministral/Ministral-3b-instruct"
    system_prompt_template = (
        "<s>[INST] {{system_message}}\n{{user_message}} [/INST] "
    )

    user_prompt_template = "<s>[INST] {{user_message}} [/INST] "
