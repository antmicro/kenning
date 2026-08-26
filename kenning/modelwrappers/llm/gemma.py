# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a model wrapper for Gemma model.
"""

import time
from functools import partial
from typing import Any, Dict, List, Optional, Union

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
        "use_sampler_loop": {
            "description": "Whether sampler loop should be used, otherwise simple loop with model.apply is used",  # noqa: E501
            "type": bool,
            "default": False,
        },
        "truncate_prompt": {
            "description": "Whether input prompt can be truncated if it is longer than specified prompt_length",  # noqa: E501
            "type": bool,
            "default": False,
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Optional[Dataset],
        from_file: bool = True,
        model_name: Optional[str] = None,
        cache_length: int = 512,
        max_out_length: int = 128,
        prompt_length: int = 128,
        use_sampler_loop: bool = False,
        truncate_prompt: bool = False,
    ):
        from gemma import gm

        self.cache_length = cache_length
        self.max_out_length = max_out_length
        self.prompt_length = prompt_length
        self.batch_size = dataset.batch_size if dataset else 1
        self.use_sampler_loop = use_sampler_loop
        self.truncate_prompt = truncate_prompt

        self.rng = None

        self.pretrained_model_uri = gm.ckpts.CheckpointPath.GEMMA3_270M_IT
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

    def load_model(self, model_path: PathOrURI):
        from gemma import gm

        self.model = gm.nn.Gemma3_270M()
        self.params = gm.ckpts.load_params(self.pretrained_model_uri)
        self.tokenizer = gm.text.Gemma3Tokenizer()

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

    def _create_simple_inference(self):
        import jax
        from gemma.gm.data._functional import pad
        from jax import lax

        @jax.jit
        def _run_single_inference(tokens_list: jax.Array):
            import jax.numpy as jnp

            generated_tokens = jnp.zeros((self.batch_size, 0), dtype=jnp.int32)
            special_tokens = self.tokenizer.special_tokens
            next_token = jnp.zeros((self.batch_size,), dtype=jnp.int32)
            for _ in range(self.max_out_length):
                is_end = jnp.all(
                    jnp.logical_or(
                        next_token == special_tokens.EOS,
                        next_token == special_tokens.END_OF_TURN,
                    )
                )

                def _infer():
                    tokens_arr = jnp.concat(
                        (tokens_list, generated_tokens), axis=-1
                    )
                    # Adjust input size to the prompt_length
                    tokens_arr = pad(
                        tokens_arr,
                        self.prompt_length,
                        truncate=self.truncate_prompt,
                        fill_value=special_tokens.PAD,
                    )

                    out = self.model.apply(
                        {"params": self.params},
                        tokens_arr,
                        return_last_only=True,
                    )
                    return jnp.argmax(out.logits, axis=-1)

                next_token = lax.cond(
                    is_end,
                    lambda: jnp.zeros((self.batch_size,), dtype=jnp.int32),
                    _infer,
                )
                generated_tokens = jnp.concat(
                    (generated_tokens, jnp.expand_dims(next_token, -1)),
                    axis=-1,
                )
            return generated_tokens

        return _run_single_inference

    def preprocess_input(self, X: List[List[str]]) -> List[List[int]]:
        inputs = super().preprocess_input(X)
        preprocessed = []
        for message in inputs[0]:
            preprocessed.append(self.tokenizer.encode(message, add_bos=True))
        return [preprocessed]

    def run_inference(self, X: List[List[int]]) -> List[List[int]]:
        import jax.numpy as jnp

        if self.use_sampler_loop:
            inference_func = partial(
                self._create_generate_tokens(), self.params, self.rng
            )
        else:
            inference_func = self._create_simple_inference()

        self.prepare_model()
        outputs = []
        for i in range(0, len(X), self.batch_size):
            predicted_tokens = inference_func(
                jnp.array(X[0][i : i + self.batch_size], dtype=jnp.int32)
            )
            outputs.extend(predicted_tokens.tolist())

        return [outputs]

    def postprocess_outputs(self, y: List[List[int]]) -> List[List[str]]:
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
                {"type": "List", "dtype": {"type": "List", "dtype": "int"}}
            ],
            "output": [
                {"type": "List", "dtype": {"type": "List", "dtype": "int"}}
            ],
            "processed_output": [{"type": "List", "dtype": "str"}],
        }

    def _create_generate_tokens(self):
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

        @jax.jit
        def generate_tokens(
            params: Any,
            rng: jax.Array,
            tokens: jax.Array,
        ) -> jax.Array:
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

        if self.use_sampler_loop:
            generate_tokens = self._create_generate_tokens()
        else:
            generate_tokens = self._create_simple_inference()

        dummy_tokens = jnp.zeros(
            (self.batch_size, self.prompt_length), dtype=jnp.int32
        )
        dummy_tokens = dummy_tokens.at[:, 0].set(
            self.tokenizer.special_tokens.BOS
        )

        KLogger.info("Lowering to StableHLO...")
        t0 = time.time()
        if self.use_sampler_loop:
            lowered = generate_tokens.lower(
                self.params, self.rng, dummy_tokens
            )
        else:
            lowered = generate_tokens.lower(dummy_tokens)
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
