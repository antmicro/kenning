# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from contextlib import contextmanager
from time import sleep
from typing import Any, Iterator, Tuple, Type
from typing import Any, Iterator, Tuple, Type

import numpy as np
import pytest

from kenning.core.dataset import Dataset
from kenning.core.measurements import MeasurementsCollector
from kenning.core.model import ModelWrapper
from kenning.core.runtime import ModelNotPreparedError, Runtime
from kenning.tests.core.conftest import (
    DatasetModelRegistry,
    DatasetModelRegistry,
    UnknownFramework,
)
from kenning.utils.class_loader import get_all_subclasses

RUNTIME_SUBCLASSES = get_all_subclasses(
    "kenning.runtimes", Runtime, raise_exception=True
)

RUNTIME_INPUTTYPES = [
    (run, inp) for run in RUNTIME_SUBCLASSES for inp in run.inputtypes
]


@contextmanager
@contextmanager
def prepare_objects(
    runtime_cls: Type[Runtime], inputtype: str, **runtime_kwargs: Any
) -> Iterator[Tuple[Runtime, Dataset, ModelWrapper]]:
    try:
        try:
            dataset, model, assets_id = DatasetModelRegistry.get(inputtype)
        except UnknownFramework:
            pytest.xfail(f"Unknown framework: {inputtype}")

        runtime = runtime_cls(**runtime_kwargs, model_path=model.model_path)
        yield runtime, dataset, model
    finally:
        DatasetModelRegistry.remove(assets_id)


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
        with prepare_objects(runtime_cls, inputtype):
            pass
        with prepare_objects(runtime_cls, inputtype):
            pass

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
        with prepare_objects(runtime_cls, inputtype) as (runtime, _, _):
            try:
                assert runtime.prepare_local()
            except NotImplementedError:
                pytest.xfail(
                    f"{runtime_cls.__name__} does not support local run"
                )
        with prepare_objects(runtime_cls, inputtype) as (runtime, _, _):
            try:
                assert runtime.prepare_local()
            except NotImplementedError:
                pytest.xfail(
                    f"{runtime_cls.__name__} does not support local run"
                )

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
        with prepare_objects(
        with prepare_objects(
            runtime_cls, inputtype, disable_performance_measurements=True
        ) as (runtime, dataset, model):
            runtime.inference_session_start()
        ) as (runtime, dataset, model):
            runtime.inference_session_start()

            assert runtime.prepare_local()
            assert runtime.prepare_local()

            for _ in range(8):
                X, _ = next(dataset)
                prepX = model.preprocess_input(X)
            for _ in range(8):
                X, _ = next(dataset)
                prepX = model.preprocess_input(X)

                assert runtime.load_input(prepX)
                runtime.run()
                assert runtime.load_input(prepX)
                runtime.run()

            assert runtime.statsmeasurements is None
            assert len(MeasurementsCollector.measurements.data) == 0
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
        with prepare_objects(
        with prepare_objects(
            runtime_cls, inputtype, disable_performance_measurements=False
        ) as (runtime, dataset, model):
            runtime.inference_session_start()
            assert runtime.prepare_local()
        ) as (runtime, dataset, model):
            runtime.inference_session_start()
            assert runtime.prepare_local()

            for _ in range(8):
                X, _ = next(dataset)
                prepX = model._preprocess_input(X)
            for _ in range(8):
                X, _ = next(dataset)
                prepX = model._preprocess_input(X)

                assert runtime.load_input(prepX)
                runtime._run()
                sleep(0.01)
                assert runtime.load_input(prepX)
                runtime._run()
                sleep(0.01)

            assert runtime.statsmeasurements is not None
            assert len(runtime.statsmeasurements.get_measurements().data) > 0
            assert runtime.statsmeasurements is not None
            assert len(runtime.statsmeasurements.get_measurements().data) > 0

            runtime.inference_session_end()
            assert runtime.statsmeasurements is None
            assert len(MeasurementsCollector.measurements.data) > 0
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
        with prepare_objects(runtime_cls, inputtype) as (runtime, _, _):
            assert runtime.prepare_model(None) is True
            assert runtime.prepare_model(b"") is True
        with prepare_objects(runtime_cls, inputtype) as (runtime, _, _):
            assert runtime.prepare_model(None) is True
            assert runtime.prepare_model(b"") is True

            with open(runtime.model_path, "rb") as model_f:
                assert b"" != model_f.read()
            with open(runtime.model_path, "rb") as model_f:
                assert b"" != model_f.read()

            with pytest.raises(Exception):
                assert runtime.prepare_model(b"Kenning") is False
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
        with prepare_objects(runtime_cls, inputtype) as (
            runtime,
            dataset,
            model,
        ):
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
        with prepare_objects(runtime_cls, inputtype) as (
            runtime,
            dataset,
            model,
        ):
            assert runtime.prepare_local()
        with prepare_objects(runtime_cls, inputtype) as (
            runtime,
            dataset,
            model,
        ):
            assert runtime.prepare_local()

            X, _ = next(dataset)
            prepX = model._preprocess_input(X)
            prepX = model.convert_input_to_bytes(prepX)
            X, _ = next(dataset)
            prepX = model._preprocess_input(X)
            prepX = model.convert_input_to_bytes(prepX)

            assert runtime.load_input_from_bytes(prepX)
            assert not runtime.load_input_from_bytes(b"")
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
        with prepare_objects(runtime_cls, inputtype) as (
            runtime,
            dataset,
            model,
        ):
            assert runtime.prepare_local()
        with prepare_objects(runtime_cls, inputtype) as (
            runtime,
            dataset,
            model,
        ):
            assert runtime.prepare_local()

            X, _ = next(dataset)
            prepX = model._preprocess_input(X)
            assert runtime.load_input(prepX)
            X, _ = next(dataset)
            prepX = model._preprocess_input(X)
            assert runtime.load_input(prepX)

            runtime.run()
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
        with prepare_objects(runtime_cls, inputtype) as (
            runtime,
            dataset,
            model,
        ):
            model_output_size = sum(
                [
                    np.prod(output["shape"])
                    * np.dtype(output["dtype"]).itemsize
                    for output in model.get_io_specification()["output"]
                ]
            )
        with prepare_objects(runtime_cls, inputtype) as (
            runtime,
            dataset,
            model,
        ):
            model_output_size = sum(
                [
                    np.prod(output["shape"])
                    * np.dtype(output["dtype"]).itemsize
                    for output in model.get_io_specification()["output"]
                ]
            )

            with pytest.raises(ModelNotPreparedError):
                runtime.upload_output(b"")
            with pytest.raises(ModelNotPreparedError):
                runtime.upload_output(b"")

            assert runtime.prepare_local()
            assert runtime.prepare_local()

            X, _ = next(dataset)
            prepX = model._preprocess_input(X)
            X, _ = next(dataset)
            prepX = model._preprocess_input(X)

            assert runtime.load_input(prepX)
            assert runtime.load_input(prepX)

            runtime.run()
            runtime.run()

            data = runtime.upload_output(b"")
            assert len(data) == model_output_size
            data = runtime.upload_output(b"")
            assert len(data) == model_output_size
