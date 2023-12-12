# Copyright (c) 2023-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides wrapper for Mistral-instruct model.

https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
"""

from typing import Optional

from kenning.modelwrappers.llm.llm import LLM


class MistralInstruct(LLM):
    """
    Wrapper for Mistral Instruct model.

    https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
    """

    pretrained_model_uri = "hf://mistralai/Mistral-7B-Instruct-v0.1"

    def message_to_instruction(
        self, user_message: str, system_message: Optional[str] = None
    ):
        if system_message is None:
            return f"[INST] {user_message} [/INST] "
        return f"[INST] {system_message}\n{user_message} [/INST] "
