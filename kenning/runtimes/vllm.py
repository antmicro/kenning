# Copyright (c) 2023-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime implementation for vLLM runtime.
"""

from typing import Any, List, Optional

from kenning.core.exceptions import (
    InputNotPreparedError,
    ModelNotLoadedError,
    ModelNotPreparedError,
)
from kenning.core.runtime import (
    Runtime,
)
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class VLLMRuntime(Runtime):
    """
    Runtime subclass that provides an API
    for inference on LLMs using vLLM.
    """

    inputtypes = [
        "safetensors-native",
        "safetensors-awq",
        "safetensors-gptq",
        "safetensors-sparsity-aware-kernel",
    ]

    arguments_structure = {
        "model_path": {
            "argparse_name": "--save-model-path",
            "description": "Path where the model directory will be uploaded",
            "type": ResourceURI,
            "default": "model_directory",
        },
        "max_tokens": {
            "description": "Maximum number of tokens to generate per "
            + "output sequence",
            "type": int,
            "default": 128,
        },
        "sparse_gptq_kernel": {
            "argparse_name": "--sparse-gptq-kernel",
            "description": "Determines whether to use custom "
            + "kernel for models optimized with GPTQ and SparseGPT "
            + "flow from GPTQSparseGPTOptimizer",
            "type": bool,
            "default": False,
        },
        "enforce_eager": {
            "argparse_name": "--enforce-eager",
            "description": "Determines whether to disable CUDA graphs",
            "type": bool,
            "default": False,
        },
        "temperature": {
            "description": "Value that controls the randomness of sampling",
            "type": float,
            "default": 1.0,
        },
        "top_k": {
            "description": "Determines how many top tokens are "
            + "considered during sampling",
            "type": int,
            "default": -1,
        },
        "max_model_len": {
            "description": "Model context length",
            "type": int,
            "nullable": True,
            "default": 4096,
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        max_tokens: int = 128,
        sparse_gptq_kernel: bool = False,
        enforce_eager: bool = False,
        temperature: float = 1.0,
        top_k: int = -1,
        max_model_len: Optional[int] = None,
        disable_performance_measurements: bool = False,
    ):
        """
        Initializes the vLLM runtime.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        max_tokens : int
            Maximum number of tokens to generate, by default 128.
        sparse_gptq_kernel: bool
            Determines whether to use custom kernel implementation for
            sparse and quantized models from GPTQSparseGPTOptimizer
        enforce_eager : bool
            Determines whether to disable CUDA graphs
        temperature : float
            Value that controls the randomness of sampling.
        top_k : int
            Determines how many top tokens are considered during sampling
        max_model_len : Optional[int]
            Model context length
        disable_performance_measurements : bool
            Disable collection and processing of performance metrics.
        """
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.sparse_gptq_kernel = sparse_gptq_kernel
        self.temperature = temperature
        self.top_k = top_k
        self.max_model_len = max_model_len
        self.enforce_eager = enforce_eager

        if self.sparse_gptq_kernel:
            from kenning_sparsity_aware_kernel.third_party.custom_vllm.quant_compressed_vllm import (  # noqa: E501
                GPTQLinearMethod,
            )
            from vllm.model_executor.layers.quantization import gptq

            gptq.GPTQLinearMethod = GPTQLinearMethod

        super().__init__(disable_performance_measurements)

    def _detect_quantization(self) -> Optional[str]:
        """
        Detects whether the model is quantized and what quantization
        method was used.

        First, the IO spec is used to detect the quantization method.
        If it does not such information, then model files are checked.
        The runtime should be able to run models that were quantized
        outside of Kenning environment.

        There is no standard way of detecting a quantization method, so
        this method checks for quantization config files in the model.

        1. GPTQ
            - `quantize_config.json`
        2. AWQ
            - `quant_config.json` (older version)
            - `quantization_config` key in `config.json` (newer version)

        If none of the files are present in the model directory, it is
        assumed that the model is not quantized.

        Raises
        ------
        ModelNotLoadedError
            If the config files are missing.

        Returns
        -------
        Optional[str]
            Name of the quantization method or None if the model is not
            quantized.
        """
        quantization = None
        if (
            self.misc_io_metadata
            and "quantization_algorithm" in self.misc_io_metadata
        ):
            quantization = self.misc_io_metadata["quantization_algorithm"]

        if quantization is not None:
            return quantization

        if (self.model_path / "quantize_config.json").is_file():
            return "GPTQ"

        if (self.model_path / "quant_config.json").is_file():
            return "AWQ"
        try:
            with open(self.model_path / "config.json", "r") as config_file:
                import json

                config = json.load(config_file)
                if "quantization_config" in config:
                    return "AWQ"
        except FileNotFoundError:
            raise KLogger.error_prepare_exception(
                "Could not find config.json file in the model directory. "
                + "Make sure the model is not corrupted",
                ModelNotLoadedError,
            )
        if quantization is not None:
            KLogger.info(
                "Detected quantization technique for vLLM runtime: "
                + {quantization}
            )
        return quantization

    def preprocess_model_to_upload(self, path: PathOrURI) -> PathOrURI:
        """
        The method preprocesses the model to be uploaded to the client and
        returns a new path to it.

        The method is used to prepare the model to be sent to the client.
        It can be used to change the model representation, for example,
        to compress it.

        Parameters
        ----------
        path : PathOrURI
            Path to the model to preprocess.

        Returns
        -------
        PathOrURI
            Path to the preprocessed model.

        Raises
        ------
        FileNotFoundError
            If the model path does not exist or if it is not a directory.
        """
        import tarfile

        if not path.is_dir():
            raise KLogger.error_prepare_exception(
                "Model path is not a directory. "
                + "Cannot compress it into a .tar file",
                FileNotFoundError,
            )

        KLogger.debug("Compressing model directory into a .tar file")

        # Compress the model directory into a .tar.gz file
        # and return a path to it
        with tarfile.open(path.with_suffix(".tar"), "w:gz") as tar:
            tar.add(path, arcname=path.name)
        path = path.with_suffix(".tar")
        return path

    def prepare_model(self, input_data: Optional[bytes]) -> bool:
        from vllm import LLM, SamplingParams

        # Models are uploaded as .tar.gz
        if input_data:
            import io
            import tarfile

            file_object = io.BytesIO(input_data)
            with tarfile.open(fileobj=file_object, mode="r:gz") as tar:
                tar_name = tar.getnames()[0]

                for member in tar.getmembers():
                    # Skip the top-level directory
                    if member.path == tar_name:
                        continue

                    # Remove the top-level directory from the path
                    new_name = member.path.removeprefix(tar_name).removeprefix(
                        "/"
                    )
                    member.name = new_name
                    tar.extract(member, path=self.model_path)

        self.sampling_params = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
        )

        self.llm = LLM(
            model=str(self.model_path),
            tokenizer=str(self.model_path),
            quantization=self._detect_quantization(),
            enforce_eager=self.enforce_eager,
            max_model_len=self.max_model_len,
        )
        return True

    def load_input(self, input_data: List[List[List[str]]]) -> bool:
        KLogger.debug(f"Loading inputs of size {len(input_data[0])}")
        if self.llm is None or self.sampling_params is None:
            raise KLogger.error_prepare_exception(
                "Prepare the model using before loading input data",
                ModelNotPreparedError,
            )
        if input_data is None or 0 == len(input_data):
            KLogger.error("Received empty input data")
            return False

        self.input_prompts = []

        self.input_prompts = input_data[0]
        return True

    def load_input_from_bytes(self, input_data: bytes) -> bool:
        KLogger.debug(f"Preparing input of size {len(input_data)}")
        input_prompts = []
        prompts = input_data.decode()

        while prompts:
            prompt_length = prompts.split(" ", 1)[0]
            if not prompt_length.isnumeric():
                raise KLogger.error_prepare_exception(
                    "Input prompt did not have length defined. "
                    + "Make sure the prompts are preprocessed properly",
                    InputNotPreparedError,
                )

            prompt = prompts[0 : len(prompt_length) + 1 + int(prompt_length)]
            input_prompts.append(prompt)
            prompts = prompts[(len(prompt_length) + 1 + int(prompt_length)) :]
        return self.load_input([[input_prompts]])

    def run(self):
        if self.llm is None or self.sampling_params is None:
            raise KLogger.error_prepare_exception(
                "Prepare the model before running inference",
                ModelNotPreparedError,
            )
        if not self.input_prompts:
            raise KLogger.error_prepare_exception(
                "Load input data before running inference",
                InputNotPreparedError,
            )
        llm_outputs = self.llm.generate(
            self.input_prompts, self.sampling_params, use_tqdm=False
        )
        self.outputs = []

        for output in llm_outputs:
            generated_text = output.outputs[0].text
            self.outputs.append(generated_text)
        self.outputs = [self.outputs]

    def extract_output(self) -> List[Any]:
        return self.outputs
