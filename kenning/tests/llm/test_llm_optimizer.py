# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Test code for checking sanity of LLM optimizers.
"""

from typing import List, Tuple, Type

import pytest

from kenning.core.optimizer import Optimizer
from kenning.modelwrappers.llm.llm import LLM
from kenning.tests.conftest import get_tmp_path
from kenning.tests.core.conftest import remove_file_or_dir
from kenning.utils.pipeline_runner import PipelineRunner
from kenning.utils.resource_manager import ResourceURI


def models_list() -> List[Type[LLM]]:
    from kenning.utils.class_loader import get_all_subclasses

    models = get_all_subclasses(
        "kenning.modelwrappers.llm",
        LLM,
        raise_exception=True,
    )

    return models


def optimizer_list() -> List[Type[Optimizer]]:
    from kenning.optimizers.awq import AWQOptimizer
    from kenning.optimizers.gptq import GPTQOptimizer

    optimizers = [
        AWQOptimizer,
        GPTQOptimizer,
    ]

    try:
        # Not installed.
        from kenning.optimizers.gptq_sparsegpt import GPTQSparseGPTOptimizer

        optimizers.append(GPTQSparseGPTOptimizer)
    except ImportError:
        pass
    return optimizers


def prepare_objects(
    modelwrapper_cls: Type[LLM], optimizer_cls: Type[Optimizer]
) -> Tuple[LLM, Optimizer]:
    # options to make testing faster
    optimizer_kwargs = {
        "batch_size": 1,
        "seqlen": 16,
        "calibration_samples": 4,
    }

    if modelwrapper_cls.__name__ == "SmolLM2":
        optimizer_kwargs["group_size"] = 64

    raw_uri = modelwrapper_cls.pretrained_model_uri
    model_path = ResourceURI(raw_uri)
    model = modelwrapper_cls(model_path, None)
    optimizer = optimizer_cls(
        None,  # No dataset
        get_tmp_path(),
        model_wrapper=model,
        **optimizer_kwargs,
    )

    return model, optimizer


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
    def test_model_optimizer_compat(self, modelwrapper_cls, optimizer_cls):
        """
        Test compilation of the optimizer given the modelwrapper.
        """
        if optimizer_cls.__name__ == "GPTQSparseGPTOptimizer":
            pytest.skip(
                "Skipping GPTQSparseGPTOptimizer test with "
                f"modelwrapper: {modelwrapper_cls.__name__}"
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
