# Copyright (c) 2023-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides wrapper for Phi-2 model.

https://huggingface.co/microsoft/phi-2
"""


from kenning.modelwrappers.llm.llm import LLM


class PHI2(LLM):
    """
    Wrapper for Phi-2 model.

    https://huggingface.co/microsoft/phi-2
    """

    pretrained_model_uri = "hf://microsoft/phi-2"

    system_prompt_template = (
        "Instruct: {{system_message}}. {{user_message}}\nOutput:"
    )

    user_prompt_template = "Instruct: {{user_message}}\nOutput:"
