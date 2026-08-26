# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a model wrapper for Gemma model.
"""

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from kenning.core.dataset import Dataset
from kenning.core.exceptions import NotSupportedError
from kenning.core.measurements import MeasurementsCollector
from kenning.modelwrappers.llm.llm import LLM
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI


class Gemma(LLM):
    """
    Model wrapper for Gemma model.

    https://github.com/google-deepmind/gemma
    """

    arguments_structure = {
        "cache_length": {
            "description": "Length of cache in the sampler loop",
            "type": int,
            "default": 512,
        },
        "max_out_length": {
            "description": "Max number of tokens in one response",
            "type": int,
            "default": 128,
        },
        "prompt_length": {
            "description": "Max length of model input",
            "type": int,
            "default": 128,
        },
        "truncate_prompt": {
            "description": "Whether input prompt can be truncated if it is longer than specified prompt_length",  # noqa: E501
            "type": bool,
            "default": False,
        },
    }

    # Default value defined by gm.ckpts.CheckpointPath.GEMMA3_270M_IT
    pretrained_model_uri = None

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Optional[Dataset],
        from_file: bool = True,
        model_name: Optional[str] = None,
        cache_length: int = 512,
        max_out_length: int = 128,
        prompt_length: int = 128,
        truncate_prompt: bool = False,
    ):
        self.cache_length = cache_length
        self.max_out_length = max_out_length
        self.prompt_length = prompt_length
        self.batch_size = dataset.batch_size if dataset else 1
        self.truncate_prompt = truncate_prompt
        self.rng = None
        self.tokenizer_prepared = False
        self.generate_tokens = None

        super().__init__(model_path, dataset, from_file, model_name)

    def message_to_instruction(
        self,
        prompt_config: Dict | str,
    ) -> str:
        from jinja2 import Template

        prompt_config = LLM._transform_prompt_config(prompt_config)
        if "system_message" in prompt_config:
            template = (
                "<start_of_turn>user\n{{system_message}}\n\n{{user_message}}"
            )
        else:
            template = "<start_of_turn>user\n{{user_message}}"
        template += "<end_of_turn>\n<start_of_turn>model\n"
        return LLM._template_to_str(
            template=Template(template), user_prompt_config=prompt_config
        )

    def prepare_tokenizer(self):
        from gemma import gm

        if self.tokenizer_prepared:
            return None
        self.tokenizer = gm.text.Gemma3Tokenizer()
        self.tokenizer_prepared = True

    def load_model(self, model_path: PathOrURI):
        from gemma import gm

        self.model = gm.nn.Gemma3_270M()
        self.params = gm.ckpts.load_params(
            self.pretrained_model_uri or gm.ckpts.CheckpointPath.GEMMA3_270M_IT
        )
        self.prepare_tokenizer()

    def prepare_model(self):
        import jax

        if self.model_prepared:
            return None

        self.rng = jax.random.PRNGKey(0)
        if self.from_file:
            self.load_model(self.model_path)
            self.model_prepared = True
        else:
            raise KLogger.error_prepare_exception(
                "LLM ModelWrapper only supports loading model from a file.",
                NotImplementedError,
            )

        self.generate_tokens = self._create_sampler_loop()

    def preprocess_input(self, X: List[List[str]]) -> List[NDArray]:
        from gemma.gm.data._functional import pad

        inputs = super().preprocess_input(X)
        self.prepare_tokenizer()
        preprocessed = []
        for message in inputs[0]:
            tok = pad(
                self.tokenizer.encode(message, add_bos=True),
                self.prompt_length,
                truncate=self.truncate_prompt,
                fill_value=self.tokenizer.special_tokens.PAD,
            )
            preprocessed.append(tok)
        return [np.asarray(preprocessed, dtype=np.int32)]

    def run_inference(self, X: List[NDArray]) -> List[NDArray]:
        import jax.numpy as jnp

        self.prepare_model()

        predicted_tokens = self.generate_tokens(
            jnp.array(X[0], dtype=jnp.int32)
        )
        outputs = predicted_tokens.tolist()

        return [outputs]

    def postprocess_outputs(self, y: List[NDArray]) -> List[List[str]]:
        self.prepare_tokenizer()
        decoded = []
        for _y in y[0]:
            MeasurementsCollector.measurements += {"tokens": [len(_y)]}
            decoded.append(self.tokenizer.decode(_y))
        return [decoded]

    @classmethod
    def _get_io_specification(cls):
        return {
            "input": [{"type": "List", "dtype": "str"}],
            "processed_input": [
                {"dtype": "int32", "shape": (-1, -1)},
            ],
            "output": [
                {"dtype": "int32", "shape": (-1, -1)},
            ],
            "processed_output": [{"type": "List", "dtype": "str"}],
            "entry_func": "jit_generate_tokens",
        }

    def _create_sampler_loop(self):
        import gemma.gm.text._prefill
        import gemma.gm.text._sampler_loop
        import gemma.gm.text._sampling
        import gemma.gm.utils._types
        import jax
        import jax.numpy as jnp
        from gemma.gm.data._functional import pad

        special_tokens = self.tokenizer.special_tokens
        end_tokens = (
            special_tokens.EOS,
            special_tokens.END_OF_TURN,
        )
        forbidden_tokens = self.tokenizer.FORBIDDEN_TOKENS
        sampling_method = gemma.gm.text._sampling.Greedy()
        params = self.params
        rng = self.rng

        @jax.jit
        def generate_tokens(tokens: jax.Array) -> jax.Array:
            inputs = gemma.gm.utils._types.Input(
                text=pad(
                    tokens,
                    self.prompt_length,
                    truncate=self.truncate_prompt,
                    fill_value=special_tokens.PAD,
                ),
                images=None,
                config=self.model.config.input_config,
            )

            init_state = gemma.gm.text._prefill.prefill(
                model=self.model,
                params=params,
                input=inputs,
                last_state=None,
                cache_length=self.cache_length,
                max_out_length=self.max_out_length,
                pad_length=None,
                rng=rng,
                sharding=None,
            )

            sampler_loop_inst = gemma.gm.text._sampler_loop.SamplerLoop(
                model=self.model,
                end_tokens=end_tokens,
                forbidden_tokens=forbidden_tokens,
                sampling=sampling_method,
                cache_length=self.cache_length,
                special_tokens=special_tokens,
            )

            final_state = sampler_loop_inst.sample(
                params=params,
                init_state=init_state,
                max_new_tokens=jnp.asarray(self.max_out_length),
                stream=False,
            )

            return final_state.predicted_tokens

        return generate_tokens

    def to_mlir(self, model_path: Union[PathOrURI, Any], **kwargs):
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_use_shardy_partitioner", False)
        self.prepare_model()

        dummy_tokens = jnp.zeros(
            (self.batch_size, self.prompt_length), dtype=jnp.int32
        )
        dummy_tokens = dummy_tokens.at[:, 0].set(
            self.tokenizer.special_tokens.BOS
        )

        KLogger.info("Lowering to StableHLO...")
        t0 = time.time()
        lowered = self.generate_tokens.lower(dummy_tokens)
        stablehlo_ir = lowered.compiler_ir(dialect="stablehlo")
        KLogger.info(f"Lowered in {time.time() - t0:.2f}s")

        stablehlo_ir_text = str(stablehlo_ir)

        return stablehlo_ir_text

    @classmethod
    def get_framework(cls):
        return "gemma"

    @classmethod
    def get_framework_version(cls):
        import gemma

        return gemma.__version__

    def train_model(self):
        raise NotSupportedError
