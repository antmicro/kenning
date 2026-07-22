# Copyright (c) 2025-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from typing import Tuple, Type

import pytest

from kenning.converters import converter_registry
from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.optimizers.model_inserter import ModelInserter
from kenning.tests.conftest import (
    get_tmp_path,
)
from kenning.tests.core.conftest import (
    DatasetModelRegistry,
    remove_file_or_dir,
)
from kenning.utils.class_loader import get_all_subclasses
from kenning.utils.pipeline_runner import PipelineRunner

OPTIMIZER_SUBCLASSES = get_all_subclasses(
    "kenning.optimizers",
    Optimizer,
    raise_exception=True,
    blacklist=["ModelInserter"],
)

EXPECTED_FAIL = [
    ("AWQOptimizer", "Ai8xCompiler"),
    ("AWQOptimizer", "ExecuTorchOptimizer"),
    ("AWQOptimizer", "GPTQOptimizer"),
    ("AWQOptimizer", "GPTQSparseGPTOptimizer"),
    ("AWQOptimizer", "IREECompiler"),
    ("AWQOptimizer", "NNIPruningOptimizer"),
    ("AWQOptimizer", "ONNXCompiler"),
    ("AWQOptimizer", "TFLiteCompiler"),
    ("AWQOptimizer", "TVMCompiler"),
    ("AWQOptimizer", "TinygradOptimizer"),
    ("GPTQOptimizer", "AWQOptimizer"),
    ("GPTQOptimizer", "Ai8xCompiler"),
    ("GPTQOptimizer", "ExecuTorchOptimizer"),
    ("GPTQOptimizer", "GPTQSparseGPTOptimizer"),
    ("GPTQOptimizer", "IREECompiler"),
    ("GPTQOptimizer", "NNIPruningOptimizer"),
    ("GPTQOptimizer", "ONNXCompiler"),
    ("GPTQOptimizer", "TFLiteCompiler"),
    ("GPTQOptimizer", "TVMCompiler"),
    ("GPTQOptimizer", "TinygradOptimizer"),
    ("GPTQSparseGPTOptimizer", "AWQOptimizer"),
    ("GPTQSparseGPTOptimizer", "Ai8xCompiler"),
    ("GPTQSparseGPTOptimizer", "ExecuTorchOptimizer"),
    ("GPTQSparseGPTOptimizer", "GPTQOptimizer"),
    ("GPTQSparseGPTOptimizer", "IREECompiler"),
    ("GPTQSparseGPTOptimizer", "NNIPruningOptimizer"),
    ("GPTQSparseGPTOptimizer", "ONNXCompiler"),
    ("GPTQSparseGPTOptimizer", "TFLiteCompiler"),
    ("GPTQSparseGPTOptimizer", "TVMCompiler"),
    ("GPTQSparseGPTOptimizer", "TinygradOptimizer"),
    ("ONNXCompiler", "Ai8xCompiler"),
    ("TFLiteCompiler", "Ai8xCompiler"),
    ("TFLiteCompiler", "NNIPruningOptimizer"),
    ("TensorFlowClusteringOptimizer", "Ai8xCompiler"),
    ("TensorFlowClusteringOptimizer", "NNIPruningOptimizer"),
    ("TensorFlowPruningOptimizer", "Ai8xCompiler"),
    ("TensorFlowPruningOptimizer", "NNIPruningOptimizer"),
    ("NNIPruningOptimizer", "Ai8xCompiler"),
    ("ONNXCompiler", "NNIPruningOptimizer"),
    ("ExecuTorchOptimizer", "Ai8xCompiler"),
    ("ExecuTorchOptimizer", "IREECompiler"),
    ("ExecuTorchOptimizer", "NNIPruningOptimizer"),
    ("ExecuTorchOptimizer", "ONNXCompiler"),
    ("ExecuTorchOptimizer", "TFLiteCompiler"),
    ("ExecuTorchOptimizer", "TVMCompiler"),
    ("ExecuTorchOptimizer", "TinygradOptimizer"),
]

SKIP = [
    ("Ai8xCompiler", "Ai8xCompiler"),
    ("TVMCompiler", "TVMCompiler"),
    ("ExecuTorchOptimizer", "ExecuTorchOptimizer"),
    ("IREECompiler", "IREECompiler"),
    ("NNIPruningOptimizer", "NNIPruningOptimizer"),
    ("TinygradOptimizer", "TinygradOptimizer"),
    # Not yet supported.
    ("GPTQOptimizer", "GPTQOptimizer"),
    ("AWQOptimizer", "AWQOptimizer"),
    ("GPTQSparseGPTOptimizer", "GPTQSparseGPTOptimizer"),
]

for optimizer in OPTIMIZER_SUBCLASSES:
    SKIP.append(("IREECompiler", optimizer.__name__))
    SKIP.append(("TinygradOptimizer", optimizer.__name__))

expected_mark = pytest.mark.xfail(reason="Expected incompatible")
skip = pytest.mark.skip(reason="No conversion available")


def prepare_objects(
    optimizer_cls1: Type[Optimizer],
    optimizer_cls2: Type[Optimizer],
    compiled_model_path: Path,
) -> Tuple[Dataset, ModelWrapper, Optimizer, Optimizer]:
    if ModelInserter in (optimizer_cls1, optimizer_cls2):
        pytest.skip("ModelInserter is not supported")

    optimizers = []
    optimizer_types = []

    for opt_cls in (optimizer_cls1, optimizer_cls2):
        optimizer = opt_cls(
            dataset=None,
            compiled_model_path=compiled_model_path,
        )

        optimizers.append(optimizer)
        optimizer_types.append(optimizer.get_framework())

    if not converter_registry.find_all_paths(
        optimizer_types[0], optimizer_types[1]
    ):
        pytest.skip("No available conversion path")

    dataset, model, _ = DatasetModelRegistry.get(optimizer_types[0])

    for optimizer, model_type in zip(optimizers, optimizer_types):
        optimizer.dataset = model.dataset
        optimizer.model_framework = model_type
        optimizer.set_input_type(model_type)
        optimizer.init()

    optimizer1, optimizer2 = optimizers
    return dataset, model, optimizer1, optimizer2


@pytest.mark.slow
class TestOptimizersCompatibility:
    @pytest.mark.compat_matrix(Optimizer, Optimizer)
    @pytest.mark.parametrize(
        "optimizer_cls1, optimizer_cls2",
        [
            pytest.param(cls1, cls2, marks=[expected_mark])
            if (cls1.__name__, cls2.__name__) in EXPECTED_FAIL
            else pytest.param(cls1, cls2, marks=[skip])
            if (cls1.__name__, cls2.__name__) in SKIP
            else (cls1, cls2)
            for cls1 in OPTIMIZER_SUBCLASSES
            for cls2 in OPTIMIZER_SUBCLASSES
        ],
    )
    def test_matrix(
        self,
        optimizer_cls1: Type[Optimizer],
        optimizer_cls2: Type[Optimizer],
    ):
        if (
            optimizer_cls1.__name__ == "TinygradOptimizer"
            or optimizer_cls2.__name__ == "TinygradOptimizer"
        ) and sys.version_info < (3, 11):
            pytest.xfail("Tinygrad is not supported on Python 3.10")
        compiled_model_path = get_tmp_path()
        dataset, model, optimizer1, optimizer2 = prepare_objects(
            optimizer_cls1,
            optimizer_cls2,
            compiled_model_path,
        )

        try:
            pipeline_runner = PipelineRunner(
                dataset=dataset,
                optimizers=[optimizer1, optimizer2],
                model_wrapper=model,
            )
            pipeline_runner.run(run_benchmarks=False)
        finally:
            compiled_model_path.unlink(missing_ok=True)
            remove_file_or_dir(model.model_path)
