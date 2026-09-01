# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Test code for checking sanity of LLM optimizers and vLLM compatibility.
"""


import pytest

from kenning.tests.core.conftest import remove_file_or_dir
from kenning.tests.llm.utils import optimizer_list
from kenning.utils.pipeline_runner import PipelineRunner


class TestLLMOptimizers:
    @pytest.mark.parametrize("optimizer_cls", optimizer_list())
    def test_optimizer_vllm_compat(self, optimizer_cls, prepare_objects):
        """
        Test optimization execution.
        """
        from kenning.modelwrappers.llm.mistral import MistralInstruct
        from kenning.runtimes.vllm import VLLMRuntime

        model, optimizer = prepare_objects(MistralInstruct, optimizer_cls)

        vllm_runtime = VLLMRuntime(
            model_path=optimizer.compiled_model_path,
            max_tokens=8,
            enforce_eager=True,
            batch_size=1,
        )

        try:
            pipeline_runner = PipelineRunner(
                dataset=None,
                optimizers=[optimizer],
                runtime=vllm_runtime,
                model_wrapper=model,
                platform=None,
            )

            pipeline_runner.run(run_benchmarks=False)

            assert optimizer.compiled_model_path.exists()
        finally:
            remove_file_or_dir(optimizer.compiled_model_path)
            remove_file_or_dir(model.model_path)
