# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a small model for testing Kenning compatibility with LLMs.

Uses: hf-internal-testing/tiny-random-LlamaForCausalLM
"""

from kenning.modelwrappers.llm.llm import LLM


class SmolLM2(LLM):
    """
    Wrapper for TinyRandom Llama. A 1.03M param LLM model.
    """

    pretrained_model_uri = "hf://HuggingFaceTB/SmolLM2-135M-Instruct"

    system_prompt_template = (
        "<|im_start|>system\n{{system_message}}<|im_end|>\n"
        "<|im_start|>user\n{{user_message}}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    user_prompt_template = (
        "<|im_start|>user\n{{user_message}}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
