# Copyright (c) 2023-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides wrapper for Phi-2 model.

https://huggingface.co/microsoft/phi-2
"""

from typing import Optional

from kenning.modelwrappers.llm.llm import LLM


class PHI2(LLM):
    """
    Wrapper for Phi-2 model.

    https://huggingface.co/microsoft/phi-2
    """

    pretrained_model_uri = "hf://microsoft/phi-2"

    def message_to_instruction(
        self, user_message: str, system_message: Optional[str] = None
    ):
        if system_message is None:
            return f"Instruct: {user_message}\nOutput:"
        return f"Instruct: {system_message}. {user_message}\nOutput:"
