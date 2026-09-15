# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import sys
import time
from typing import Dict, Generator, List, Tuple
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from kenning.core.dataset import Dataset
from kenning.core.exceptions import (
    ModelNotPreparedError,
)
from kenning.core.inferenceloop import InferenceLoop
from kenning.core.measurements import Measurements
from kenning.core.platform import Platform
from kenning.core.protocol import Protocol
from kenning.core.runtime import Runtime
from kenning.dataconverters.modelwrapper_dataconverter import (
    ModelWrapperDataConverter,
)
from kenning.inferenceloops.anomaly_realtime import (
    AnomalyDetectionInferenceLoop,
)
from kenning.inferenceloops.local_sequential import (
    LocalSequentialInferenceLoop,
)
from kenning.inferenceloops.remote_sequential import (
    RemoteSequentialInferenceLoop,
)
from kenning.inferenceloops.sensor_realtime import (
    SensorRealtimeInferenceLoop,
)
from kenning.platforms.simulatable_platform import SimulatablePlatform
from kenning.platforms.zephyr import ZephyrPlatform
from kenning.tests.core.conftest import (
    DatasetModelRegistry,
)

PLATFORM_CONFIG = {
    "LOCAL": (LocalSequentialInferenceLoop, "max78002evkit/max78002/m4"),
    "REMOTE": (RemoteSequentialInferenceLoop, "nvidia_rtx_4090"),
    "SENSOR": (SensorRealtimeInferenceLoop, "max32690evkit/max32690/m4"),
    "ANOMALY": (AnomalyDetectionInferenceLoop, "max32690evkit/max32690/m4"),
}


@pytest.fixture
def renode_machine_mock() -> MagicMock:
    machine = MagicMock()
    machine.sysbus.i2c.adxl345.internal = Mock(spec=type("ADXL345", (), {}))
    return machine


@pytest.fixture
def system_mock(monkeypatch):
    mock = MagicMock()
    monkeypatch.setitem(sys.modules, "System", mock)
    return mock


@pytest.fixture
def platform_factory(system_mock, renode_machine_mock):
    def _create_mock(platform_name: str, platform_cls):
        mock = Mock(spec=platform_cls)
        config = {
            "name": platform_name,
            "sensors": ["i2c.adxl345"],
            "sensors_frequency": 100.0,
            "machine": renode_machine_mock,
            "needs_protocol": True,
        }

        mock.configure_mock(**config)
        mock.get_time.side_effect = time.perf_counter
        return mock

    return _create_mock


@pytest.fixture
def protocol_factory():
    def _protocol_mock(output_shape: np.ndarray, output_dtype: np.dtype):
        mock = Mock(spec=Protocol)
        mock.upload_io_specification.return_value = True
        mock.upload_model.return_value = True
        mock.download_statistics.return_value = {}
        mock.upload_input.return_value = True
        mock.request_processing.return_value = True
        mock.download_output.return_value = (
            True,
            np.ones(output_shape, dtype=output_dtype),
        )
        mock.deduce_data_converter_from_io_spec.return_value = None
        return mock

    return _protocol_mock


@pytest.fixture
def runtime_factory():
    def _runtime_mock(output_shape: np.ndarray, output_dtype: np.dtype):
        mock = Mock(spec=Runtime)
        mock.load_input.return_value = True
        mock.extract_output.return_value = [
            np.ones(output_shape, dtype=output_dtype)
        ]
        mock.get_available_ram.return_value = 1024 * 1024
        return mock

    return _runtime_mock


@pytest.fixture
def prepare_objects(
    request, runtime_factory, platform_factory, protocol_factory
) -> Generator[Tuple[Dataset, InferenceLoop], None, None]:
    assets_id = None
    inference_loop_cls, platform_name = request.param
    try:
        dataset, model, assets_id = DatasetModelRegistry.get("tvm")
        dataconverter = ModelWrapperDataConverter(model)

        output_layer = model.get_io_specification()["output"][0]
        runtime = runtime_factory(output_layer["shape"], output_layer["dtype"])
        platform = Platform(platform_name)

        loop_kwargs = {
            "dataset": dataset,
            "dataconverter": dataconverter,
            "model_wrapper": model,
            "runtime": runtime,
            "platform": platform,
        }

        if inference_loop_cls is RemoteSequentialInferenceLoop:
            protocol = protocol_factory(
                output_layer["shape"], output_layer["dtype"]
            )
            loop_kwargs.update(
                {
                    "model_path": str(model.model_path),
                    "protocol": protocol,
                }
            )
        elif inference_loop_cls is AnomalyDetectionInferenceLoop:
            protocol = protocol_factory(
                output_layer["shape"], output_layer["dtype"]
            )
            platform = platform_factory(platform_name, SimulatablePlatform)
            loop_kwargs.update(
                {
                    "platform": platform,
                    "protocol": protocol,
                }
            )

        elif inference_loop_cls is (SensorRealtimeInferenceLoop):
            protocol = protocol_factory(
                output_layer["shape"], output_layer["dtype"]
            )
            platform = platform_factory(platform_name, ZephyrPlatform)
            loop_kwargs.update(
                {
                    "platform": platform,
                    "protocol": protocol,
                }
            )

        inference_loop = inference_loop_cls(**loop_kwargs)
        yield dataset, inference_loop
    finally:
        if assets_id is not None:
            DatasetModelRegistry.remove(assets_id)


@pytest.fixture
def sensor_measurements() -> Dict[str, List]:
    spec = {
        "results_scored": [
            [1.0399, [0], "0.99991995"],
            [1.0661, [1], "0.9998966"],
            [1.0921, [0], "0.9974551"],
        ],
        "samples": [
            [
                1.02472941,
                [
                    [-0.0344, 0.0321, -0.0121, 0.0089, 0.0016, -0.0081],
                    [1.0, 0.0],
                ],
            ],
            [
                1.05073709,
                [
                    [-0.037, 0.0342, -0.0128, 0.0082, 0.0027, -0.0095],
                    [1.0, 0.0],
                ],
            ],
            [
                1.07674477,
                [
                    [-0.0392, 0.0363, 0.0015, 0.0116, -0.0069, -0.0851],
                    [1.0, 0.0],
                ],
            ],
        ],
    }
    return spec


@pytest.mark.parametrize(
    "prepare_objects",
    PLATFORM_CONFIG.values(),
    indirect=True,
)
def test_prepare(prepare_objects):
    _, inference_loop = prepare_objects

    try:
        inference_loop._prepare()
    finally:
        inference_loop._cleanup()


@pytest.mark.parametrize(
    "prepare_objects",
    [
        PLATFORM_CONFIG["LOCAL"],
        PLATFORM_CONFIG["REMOTE"],
    ],
    indirect=True,
)
def test_inference_step(prepare_objects):
    dataset, inference_loop = prepare_objects
    try:
        inference_loop._prepare()
        iterator = dataset.iter_test()

        if len(iterator) == 0:
            pytest.skip("Unable to test `_inference_step()` method.")

        for X, _ in iterator:
            prepX = inference_loop._preprocess(X)
            _, _ = inference_loop._inference_step(prepX)
            break
    finally:
        inference_loop._cleanup()


@pytest.mark.parametrize(
    "prepare_objects",
    PLATFORM_CONFIG.values(),
    indirect=True,
)
def test_run_loop(prepare_objects):
    _, inference_loop = prepare_objects
    measurements = Measurements()

    try:
        inference_loop._prepare()
        inference_loop._run_loop(measurements)
    finally:
        inference_loop._cleanup()


@pytest.mark.parametrize(
    "prepare_objects",
    [
        PLATFORM_CONFIG["LOCAL"],
    ],
    indirect=True,
)
def test_unsuccessful_run_loop(prepare_objects):
    _, inference_loop = prepare_objects
    measurements = Measurements()
    inference_loop._runtime.load_input.side_effect = ModelNotPreparedError

    try:
        inference_loop._prepare()

        with pytest.raises(ModelNotPreparedError):
            inference_loop._run_loop(measurements)
    finally:
        inference_loop._cleanup()


@pytest.mark.parametrize(
    "prepare_objects",
    [
        PLATFORM_CONFIG["ANOMALY"],
    ],
    indirect=True,
)
def test_compute_metrics(prepare_objects, sensor_measurements):
    measurements = Measurements()
    data = sensor_measurements
    for key, value in data.items():
        measurements += {key: value}

    _, inference_loop = prepare_objects
    try:
        inference_loop._prepare()
    finally:
        inference_loop._cleanup()
        inference_loop._compute_metrics(measurements)

    assert measurements.data["anomaly_metrics"]["accuracy"] == 0.5
    assert measurements.data["anomaly_metrics"]["f1"] == 0.0
