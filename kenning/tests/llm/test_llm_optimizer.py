# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Test code for checking sanity of LLM optimizers.
"""


import pytest

from kenning.tests.core.conftest import remove_file_or_dir
from kenning.tests.llm.utils import models_list, optimizer_list
from kenning.utils.pipeline_runner import PipelineRunner


class TestLLMOptimizers:
    @pytest.mark.parametrize("optimizer_cls", optimizer_list())
    def test_optimizer_version(self, optimizer_cls):
        """
        Sanity check to ensure that the LLM optimizer class is valid.
        """
        assert optimizer_cls is not None
        assert optimizer_cls.get_framework_version() != ""

    @pytest.mark.parametrize(
        "optimizer_cls",
        optimizer_list(),
    )
    @pytest.mark.parametrize("modelwrapper_cls", models_list())
    def test_model_optimizer_compat(
        self, modelwrapper_cls, optimizer_cls, prepare_objects
    ):
        """
        Test compilation of the optimizer given the modelwrapper.
        """
        if optimizer_cls.__name__ == "GPTQSparseGPTOptimizer":
            if modelwrapper_cls.__name__ in [
                "Llama",
                "SmolLM2",
            ]:
                pytest.xfail(
                    f"{modelwrapper_cls.__name__} is not compatible"
                    " with GPTQSparseGPTOptimizer."
                )

        model, optimizer = prepare_objects(modelwrapper_cls, optimizer_cls)

        try:
            pipeline_runner = PipelineRunner(
                dataset=None,
                optimizers=[optimizer],
                model_wrapper=model,
                platform=None,
            )
            pipeline_runner.run(run_benchmarks=False)
            assert optimizer.compiled_model_path.exists()
        finally:
            remove_file_or_dir(optimizer.compiled_model_path)
            remove_file_or_dir(model.model_path)
