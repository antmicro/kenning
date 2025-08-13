# Copyright (c) 2023-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides base methods for using LLMs in Kenning.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from jinja2 import Template
from transformers import __version__ as transformers_version

from kenning.core.dataset import Dataset
from kenning.core.exceptions import MissingUserMessage
from kenning.core.model import ModelWrapper
from kenning.datasets.cnn_dailymail import CNNDailymailDataset
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI


class LLM(ModelWrapper, ABC):
    """
    Base model wrapper for LLMs.
    """

    default_dataset = CNNDailymailDataset

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Optional[Dataset],
        from_file: bool = True,
        model_name: Optional[str] = None,
    ):
        super().__init__(model_path, dataset, from_file, model_name)

    @staticmethod
    def _transform_prompt_config(prompt_config: Dict | str) -> Dict:
        if (
            isinstance(prompt_config, Dict)
            and "user_message" not in prompt_config
        ):
            raise MissingUserMessage(
                "`user_message` key is missing in the "
                "`prompt_config` dictionary."
            )
        elif isinstance(prompt_config, str):
            prompt_config = {"user_message": prompt_config}
        return prompt_config

    @staticmethod
    def _template_to_str(
        template: Template,
        user_prompt_config: Dict,
        default_prompt_config: Dict = {},
    ) -> str:
        prompt_config = default_prompt_config | user_prompt_config
        return template.render(prompt_config)

    @abstractmethod
    def message_to_instruction(
        self,
        prompt_config: Dict | str,
    ) -> str:
        """
        Generate a textual prompt based on a prompt template
        and template configuration.

        Convert the provided `user_message` to a prompt that can be
        passed to a model. Format of the prompt may differ depending
        on the model architecture.

        Parameters
        ----------
        prompt_config : Dict | str
            Dictionary key-value mapping for Jinja2 template.
            `user_message` key is the only one required for all models.
            Also, str is allowed if solely a user message is provided.

        Returns
        -------
        str
            Formatted prompt for a given model.
        """
        ...

    def load_model(self, model_path: PathOrURI):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(str(model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    def prepare_model(self):
        if self.model_prepared:
            return None

        if self.from_file:
            self.load_model(self.model_path)
            self.model_prepared = True
        else:
            raise KLogger.error_prepare_exception(
                "LLM ModelWrapper only supports loading model from a file.",
                NotImplementedError,
            )

    def save_model(self, model_path: PathOrURI):
        self.prepare_model()

        self.model.save_pretrained(str(model_path))
        self.tokenizer.save_pretrained(str(model_path))

    def convert_input_to_bytes(self, inputdata: List[List[str]]) -> bytes:
        """
        Converts the input returned by the ``preprocess_input`` method
        to bytes.

        Before the conversion is done, every message is prefixed with
        its length, so that the model can properly decode the input.

        Parameters
        ----------
        inputdata : List[List[str]]
            The preprocessed inputs.

        Returns
        -------
        bytes
            Input data as byte stream.
        """
        conversations = []
        for message in inputdata:
            conversations.append(f"{len(message)} {message}")

        data = bytes()
        for message in conversations:
            data += message.encode()
        return data

    def preprocess_input(self, X: List[List[str]]) -> List[List[str]]:
        """
        Preprocesses the inputs for a given model before inference by
        adding a system message to the user message.

        Parameters
        ----------
        X : List[List[str]]
            The input data from the Dataset object.

        Returns
        -------
        List[List[str]]
            The list of prompts for the model along with system messages.
        """
        conversations = []
        for message in X[0]:
            prompt_config = {"user_message", message}
            if hasattr(self.dataset, "system_message"):
                prompt_config["system_message"] = self.dataset.system_message
            message = self.message_to_instruction(prompt_config)
            conversations.append(message)
        return [conversations]

    def convert_output_from_bytes(self, outputdata: bytes) -> List[List[str]]:
        """
        Converts the output from bytes to a list of strings.

        It is assumed that the output is a sequence of strings, each
        preceded by its length.

        Parameters
        ----------
        outputdata : bytes
            The output data from the model.

        Returns
        -------
        List[List[str]]
            The list of output strings.

        Raises
        ------
        RuntimeError
            If the output data is not properly formatted, ie. the length
            of the output does not precede the output string or its
            value does not match the length of the string.
        """
        result = []
        output = outputdata.decode()

        while output:
            prompt_length = output.split(" ", 1)[0]
            if not prompt_length.isnumeric():
                raise KLogger.error_prepare_exception(
                    "Output did not have length defined. "
                    + "Make sure the outputs are sent properly",
                    RuntimeError,
                )

            if len(prompt_length) + 1 + int(prompt_length) > len(output):
                raise KLogger.error_prepare_exception(
                    "Output length is greater than the output. "
                    + "Make sure the outputs are sent properly",
                    RuntimeError,
                )

            prompt = output[0 : len(prompt_length) + 1 + int(prompt_length)]
            result.append(prompt)
            output = output[(len(prompt_length) + 1 + int(prompt_length)) :]

        if len(output) != 0:
            raise KLogger.error_prepare_exception(
                "The output was not fully consumed. "
                + "Make sure the outputs are sent properly",
                RuntimeError,
            )

        return [result]

    @classmethod
    def _get_io_specification(cls):
        return {
            "input": [{"type": "List", "dtype": "str"}],
            "output": [{"type": "List", "dtype": "str"}],
        }

    @classmethod
    def derive_io_spec_from_json_params(
        cls, json_dict: Dict
    ) -> Dict[str, List[Dict]]:
        return cls._get_io_specification()

    def get_io_specification_from_model(self):
        return self._get_io_specification()

    def run_inference(self, X: List[List[str]]) -> List[List[str]]:
        self.prepare_model()

        if self.tokenizer.pad_token is None:
            KLogger.warning(
                "The tokenizer does not have a padding token. "
                + "Defaulting to eos_token",
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

        outputs = []
        for x in X[0]:
            inputs = self.tokenizer(x, return_tensors="pt")
            generated_ids = self.model.generate(
                inputs.input_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                max_length=200,
            )
            output = self.tokenizer.batch_decode(generated_ids)
            outputs.append(output[0])

        return [outputs]

    def save_to_onnx(self):
        raise NotImplementedError

    def get_framework_and_version(self) -> Tuple[str, str]:
        return "transformers", transformers_version

    @classmethod
    def get_output_formats(cls):
        return ["safetensors-native", "safetensors-awq"]
