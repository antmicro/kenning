from pathlib import Path
from typing import Tuple, Type

import pytest

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
    "kenning.optimizers", Optimizer, raise_exception=True
)


EXPECTED_FAIL = [
    ("NNIPruningOptimizer", "Ai8xCompiler"),
    ("NNIPruningOptimizer", "NNIPruningOptimizer"),
    ("NNIPruningOptimizer", "ONNXCompiler"),
    ("NNIPruningOptimizer", "TVMCompiler"),
    ("ONNXCompiler", "NNIPruningOptimizer"),
]
expected_mark = pytest.mark.xfail(reason="Expected incompatible")


def prepare_objects(
    optimizer_cls1: Type[Optimizer],
    optimizer_cls2: Type[Optimizer],
    compiled_model_path: Path,
) -> Tuple[Dataset, ModelWrapper, Optimizer, Optimizer]:
    if ModelInserter in (optimizer_cls1, optimizer_cls2):
        pytest.skip("ModelInserter is not supported")

    try:
        model_type_between = Optimizer.consult_model_type(
            optimizer_cls2, optimizer_cls1
        )
    except ValueError:
        pytest.skip("Blocks do not match")

    model_type_input = list(optimizer_cls1.inputtypes.keys())[0]
    dataset, model, _ = DatasetModelRegistry.get(model_type_input)

    optimizers = []
    for cls, model_type in [
        (optimizer_cls1, model_type_input),
        (optimizer_cls2, model_type_between),
    ]:
        optimizer = cls(
            model.dataset,
            compiled_model_path,
            model_framework=model_type,
        )
        optimizer.init()
        optimizers.append(optimizer)

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
