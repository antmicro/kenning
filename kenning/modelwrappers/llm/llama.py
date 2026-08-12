# Copyright (c) 2023-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides wrapper for Llama 2 model.

https://huggingface.co/meta-llama
"""

from typing import Optional

from kenning.core.dataset import Dataset
from kenning.modelwrappers.llm.llm import LLM
from kenning.utils.resource_manager import PathOrURI


class Llama(LLM):
    """
    Wrapper for Llama2-chat models created by Meta.

    https://huggingface.co/meta-llama
    """

    system_prompt_template = (
        "<s>[INST] <<SYS>>\n{{system_message}}\n<</SYS>>\n\n"
        "{{user_message}} [/INST]"
    )
    user_prompt_template = "<s>[INST] {user_message} [/INST] "

    pretrained_model_uri = "hf://TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    arguments_structure = {
        "model_version": {
            "description": "Version of the model to be used",
            "type": str,
            "enum": ["1.1B", "7B", "13B", "70B"],
            "default": "1.1B",
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Optional[Dataset],
        from_file: bool = True,
        model_name: Optional[str] = None,
        model_version: str = "1.1B",
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

        if model_version == "1.1B":
            self.pretrained_model_uri = (
                "hf://TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            )
        else:
            self.pretrained_model_uri = (
                f"hf://meta-llama/Llama-2-{self.model_version}-chat-hf"
            )

        super().__init__(model_path, dataset, from_file, model_name)
