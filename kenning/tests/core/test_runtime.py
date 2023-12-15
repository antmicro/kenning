# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from time import sleep
from typing import Any, Tuple, Type

import numpy as np
import pytest

from kenning.core.dataset import Dataset
from kenning.core.measurements import MeasurementsCollector
from kenning.core.model import ModelWrapper
from kenning.core.runtime import ModelNotPreparedError, Runtime
from kenning.runtimes.renode import RenodeRuntime
from kenning.tests.core.conftest import (
    UnknownFramework,
    get_default_dataset_model,
)
from kenning.utils.class_loader import get_all_subclasses

RUNTIME_SUBCLASSES = get_all_subclasses(
    "kenning.runtimes", Runtime, raise_exception=True
)

RUNTIME_INPUTTYPES = [
    (run, inp) for run in RUNTIME_SUBCLASSES for inp in run.inputtypes
]


def prepare_objects(
    runtime_cls: Type[Runtime], inputtype: str, **runtime_kwargs: Any
) -> Tuple[Runtime, Dataset, ModelWrapper]:
    try:
        dataset, model = get_default_dataset_model(inputtype)
    except UnknownFramework:
        pytest.xfail(f"Unknown framework: {inputtype}")

    if runtime_cls is RenodeRuntime:
        pytest.xfail("RenodeRuntime is not a regular runtime")
    else:
        runtime = runtime_cls(**runtime_kwargs, model_path=model.model_path)

    return runtime, dataset, model


class TestRuntime:
    @pytest.mark.xdist_group(name="use_resources")
    @pytest.mark.parametrize(
        "runtime_cls,inputtype",
        [
            pytest.param(
                runtime_cls,
                inputtype,
                marks=[
                    pytest.mark.dependency(
                        name=f"test_initializer[{runtime_cls.__name__}-{inputtype}]"
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestRuntime_{runtime_cls.__name__}"
                    ),
                ],
            )
            for runtime_cls, inputtype in RUNTIME_INPUTTYPES
        ],
    )
    def test_initializer(self, runtime_cls: Type[Runtime], inputtype: str):
        """
        Tests runtime initialization.
        """
        _ = prepare_objects(runtime_cls, inputtype)

    @pytest.mark.xdist_group(name="use_resources")
    @pytest.mark.parametrize(
        "runtime_cls,inputtype",
        [
            pytest.param(
                runtime_cls,
                inputtype,
                marks=[
                    pytest.mark.dependency(
                        name=f"test_prepare_local[{runtime_cls.__name__}-{inputtype}]",
                        depends=[
                            f"test_initializer[{runtime_cls.__name__}-{inputtype}]"
                        ],
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestRuntime_{runtime_cls.__name__}"
                    ),
                ],
            )
            for runtime_cls, inputtype in RUNTIME_INPUTTYPES
        ],
    )
    def test_prepare_local(self, runtime_cls: Type[Runtime], inputtype: str):
        """
        Tests the `preprocess_input` method.
        """
        runtime, _, _ = prepare_objects(runtime_cls, inputtype)

        try:
            assert runtime.prepare_local()
        except NotImplementedError:
            pytest.xfail(f"{runtime_cls.__name__} does not support local run")

    @pytest.mark.xdist_group(name="use_resources")
    @pytest.mark.parametrize(
        "runtime_cls,inputtype",
        [
            pytest.param(
                runtime_cls,
                inputtype,
                marks=[
                    pytest.mark.dependency(
                        depends=[
                            f"test_initializer[{runtime_cls.__name__}-{inputtype}]"
                        ]
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestRuntime_{runtime_cls.__name__}"
                    ),
                ],
            )
            for runtime_cls, inputtype in RUNTIME_INPUTTYPES
        ],
    )
    def test_inference_session_no_stats(
        self, runtime_cls: Type[Runtime], inputtype: str
    ):
        """
        Tests the inference session without statistics collection.
        """
        runtime, dataset, model = prepare_objects(
            runtime_cls, inputtype, disable_performance_measurements=True
        )
        runtime.inference_session_start()

        assert runtime.prepare_local()

        for _ in range(8):
            X, _ = next(dataset)
            prepX = model.preprocess_input(X)

            assert runtime.load_input(prepX)
            runtime.run()

        assert runtime.statsmeasurements is None
        assert len(MeasurementsCollector.measurements.data) == 0

        runtime.inference_session_end()
        MeasurementsCollector.clear()

    @pytest.mark.xdist_group(name="use_resources")
    @pytest.mark.parametrize(
        "runtime_cls,inputtype",
        [
            pytest.param(
                runtime_cls,
                inputtype,
                marks=[
                    pytest.mark.dependency(
                        depends=[
                            f"test_initializer[{runtime_cls.__name__}-{inputtype}]"
                        ]
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestRuntime_{runtime_cls.__name__}"
                    ),
                ],
            )
            for runtime_cls, inputtype in RUNTIME_INPUTTYPES
        ],
    )
    def test_inference_session_stats(
        self, runtime_cls: Type[Runtime], inputtype: str
    ):
        """
        Tests the inference session statistics collection.
        """
        runtime, dataset, model = prepare_objects(
            runtime_cls, inputtype, disable_performance_measurements=False
        )
        runtime.inference_session_start()

        assert runtime.prepare_local()

        for _ in range(8):
            X, _ = next(dataset)
            prepX = model._preprocess_input(X)

            assert runtime.load_input(prepX)
            runtime._run()
            sleep(0.01)

        assert runtime.statsmeasurements is not None
        assert len(runtime.statsmeasurements.get_measurements().data) > 0

        runtime.inference_session_end()
        assert runtime.statsmeasurements is None
        assert len(MeasurementsCollector.measurements.data) > 0

        MeasurementsCollector.clear()

    @pytest.mark.xdist_group(name="use_resources")
    @pytest.mark.parametrize(
        "runtime_cls,inputtype",
        [
            pytest.param(
                runtime_cls,
                inputtype,
                marks=[
                    pytest.mark.dependency(
                        depends=[
                            f"test_initializer[{runtime_cls.__name__}-{inputtype}]"
                        ]
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestRuntime_{runtime_cls.__name__}"
                    ),
                ],
            )
            for runtime_cls, inputtype in RUNTIME_INPUTTYPES
        ],
    )
    def test_prepare_model(self, runtime_cls: Type[Runtime], inputtype: str):
        """
        Tests the `prepare_model` method.
        """
        runtime, _, _ = prepare_objects(runtime_cls, inputtype)

        assert runtime.prepare_model(None) is True

        assert runtime.prepare_model(b"") is True

        with open(runtime.model_path, "rb") as model_f:
            assert b"" != model_f.read()

        with pytest.raises(Exception):
            assert runtime.prepare_model(b"Kenning") is False

    @pytest.mark.xdist_group(name="use_resources")
    @pytest.mark.parametrize(
        "runtime_cls,inputtype",
        [
            pytest.param(
                runtime_cls,
                inputtype,
                marks=[
                    pytest.mark.dependency(
                        depends=[
                            f"test_initializer[{runtime_cls.__name__}-{inputtype}]"
                        ]
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestRuntime_{runtime_cls.__name__}"
                    ),
                ],
            )
            for runtime_cls, inputtype in RUNTIME_INPUTTYPES
        ],
    )
    def test_load_input(self, runtime_cls: Type[Runtime], inputtype: str):
        """
        Tests the `load_input` method.
        """
        runtime, dataset, model = prepare_objects(runtime_cls, inputtype)

        assert runtime.prepare_local()

        X, _ = next(dataset)
        prepX = model._preprocess_input(X)

        assert runtime.load_input(prepX)

        assert not runtime.load_input([])

    @pytest.mark.xdist_group(name="use_resources")
    @pytest.mark.parametrize(
        "runtime_cls,inputtype",
        [
            pytest.param(
                runtime_cls,
                inputtype,
                marks=[
                    pytest.mark.dependency(
                        depends=[
                            f"test_initializer[{runtime_cls.__name__}-{inputtype}]"
                        ]
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestRuntime_{runtime_cls.__name__}"
                    ),
                ],
            )
            for runtime_cls, inputtype in RUNTIME_INPUTTYPES
        ],
    )
    def test_load_input_from_bytes(
        self, runtime_cls: Type[Runtime], inputtype: str
    ):
        """
        Tests the `load_input` method.
        """
        runtime, dataset, model = prepare_objects(runtime_cls, inputtype)

        assert runtime.prepare_local()

        X, _ = next(dataset)
        prepX = model._preprocess_input(X)
        prepX = model.convert_input_to_bytes(prepX)

        assert runtime.load_input_from_bytes(prepX)

        assert not runtime.load_input_from_bytes(b"")

    @pytest.mark.xdist_group(name="use_resources")
    @pytest.mark.parametrize(
        "runtime_cls,inputtype",
        [
            pytest.param(
                runtime_cls,
                inputtype,
                marks=[
                    pytest.mark.dependency(
                        name=f"test_run[{runtime_cls.__name__}-{inputtype}]",
                        depends=[
                            f"test_prepare_local[{runtime_cls.__name__}-{inputtype}]"
                        ],
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestRuntime_{runtime_cls.__name__}"
                    ),
                ],
            )
            for runtime_cls, inputtype in RUNTIME_INPUTTYPES
        ],
    )
    def test_run(self, runtime_cls: Type[Runtime], inputtype: str):
        """
        Tests the `run_locally` method.
        """
        runtime, dataset, model = prepare_objects(runtime_cls, inputtype)

        assert runtime.prepare_local()

        X, _ = next(dataset)
        prepX = model._preprocess_input(X)

        assert runtime.load_input(prepX)

        runtime.run()

    @pytest.mark.xdist_group(name="use_resources")
    @pytest.mark.parametrize(
        "runtime_cls,inputtype",
        [
            pytest.param(
                runtime_cls,
                inputtype,
                marks=[
                    pytest.mark.dependency(
                        depends=[
                            f"test_run[{runtime_cls.__name__}-{inputtype}]"
                        ]
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestRuntime_{runtime_cls.__name__}"
                    ),
                ],
            )
            for runtime_cls, inputtype in RUNTIME_INPUTTYPES
        ],
    )
    def test_upload_output(self, runtime_cls: Type[Runtime], inputtype: str):
        """
        Tests the `upload_output` method.
        """
        runtime, dataset, model = prepare_objects(runtime_cls, inputtype)

        model_output_size = sum(
            [
                np.prod(output["shape"]) * np.dtype(output["dtype"]).itemsize
                for output in model.get_io_specification()["output"]
            ]
        )

        with pytest.raises(ModelNotPreparedError):
            runtime.upload_output(b"")

        assert runtime.prepare_local()

        X, _ = next(dataset)
        prepX = model._preprocess_input(X)

        assert runtime.load_input(prepX)

        runtime.run()

        data = runtime.upload_output(b"")
        assert len(data) == model_output_size
