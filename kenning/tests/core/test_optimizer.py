# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from typing import Iterator, Tuple, Type

import pytest

from kenning.core.model import ModelWrapper
from kenning.core.optimizer import ConversionError, Optimizer
from kenning.tests.conftest import get_tmp_path
from kenning.tests.core.conftest import (
    DatasetModelRegistry,
    UnknownFramework,
)
from kenning.utils.class_loader import get_all_subclasses

OPTIMIZER_SUBCLASSES = get_all_subclasses(
    "kenning.optimizers", Optimizer, raise_exception=True
)

OPTIMIZER_INPUTTYPES = [
    (opt, inp) for opt in OPTIMIZER_SUBCLASSES for inp in opt.inputtypes
]


@contextmanager
def prepare_objects(
    opt_cls: Type[Optimizer], inputtype: str
) -> Iterator[Tuple[Optimizer, ModelWrapper]]:
    assets_id = None
    try:
        dataset, model, assets_id = DatasetModelRegistry.get(inputtype)
        compiled_model_path = get_tmp_path()
        optimizer = opt_cls(
            dataset,
            compiled_model_path,
            model_framework=inputtype,
        )
        optimizer.set_input_type(inputtype)
        yield optimizer, model
    except UnknownFramework:
        pytest.xfail(f"Unknown framework: {inputtype}")
    finally:
        if assets_id is not None:
            DatasetModelRegistry.remove(assets_id)


class TestOptimizer:
    @pytest.mark.xdist_group(name="use_resources")
    @pytest.mark.parametrize(
        "opt_cls,inputtype",
        [
            pytest.param(
                opt_cls,
                inputtype,
                marks=[
                    pytest.mark.dependency(
                        name=f"test_initializer[{opt_cls.__name__},{inputtype}]"
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestOptimizer_{opt_cls.__name__}"
                    ),
                ],
            )
            for opt_cls, inputtype in OPTIMIZER_INPUTTYPES
        ],
    )
    def test_initializer(self, opt_cls: Type[Optimizer], inputtype: str):
        """
        Tests optimizer initialization.
        """
        with prepare_objects(opt_cls, inputtype):
            pass

    @pytest.mark.xdist_group(name="use_resources")
    @pytest.mark.parametrize(
        "opt_cls,inputtype",
        [
            pytest.param(
                opt_cls,
                inputtype,
                marks=[
                    pytest.mark.dependency(
                        depends=[
                            f"test_initializer[{opt_cls.__name__},{inputtype}]"
                        ]
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestOptimizer_{opt_cls.__name__}"
                    ),
                    pytest.mark.skipif(
                        opt_cls.__name__ == "NNIPruningOptimizer"
                        and inputtype == "onnx",
                        reason="Pruning and fine-tuning models like YOLO"
                        " takes too much time",
                    ),
                ],
            )
            for opt_cls, inputtype in OPTIMIZER_INPUTTYPES
        ],
    )
    def test_compilation(self, opt_cls: Type[Optimizer], inputtype: str):
        """
        Tests optimizer compilation.
        """
        with prepare_objects(opt_cls, inputtype) as (optimizer, model):
            try:
                optimizer.init()
                optimizer.compile(model.model_path)
                assert optimizer.compiled_model_path.exists()
            except ConversionError as e:
                pytest.xfail(f"conversion error {e}")

    @pytest.mark.xdist_group(name="use_resources")
    @pytest.mark.parametrize(
        "opt_cls,inputtype",
        [
            pytest.param(
                opt_cls,
                inputtype,
                marks=[
                    pytest.mark.dependency(
                        depends=[
                            f"test_initializer[{opt_cls.__name__},{inputtype}]"
                        ]
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestOptimizer_{opt_cls.__name__}"
                    ),
                ],
            )
            for opt_cls, inputtype in OPTIMIZER_INPUTTYPES
        ],
    )
    def test_get_framework_and_version(
        self, opt_cls: Type[Optimizer], inputtype: str
    ):
        """
        Tests `get_framework_and_version` method.
        """
        with prepare_objects(opt_cls, inputtype) as (optimizer, _):
            assert optimizer.get_framework_and_version() is not None
