# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import Mock

import pytest

from kenning.core.dataconverter import DataConverter
from kenning.core.dataset import Dataset
from kenning.core.measurements import MeasurementsCollector
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.core.protocol import Protocol, RequestFailure
from kenning.core.runtime import Runtime
from kenning.utils.pipeline_runner import PipelineRunner


@pytest.fixture
def protocol_mock():
    return Mock(spec=Protocol)


@pytest.fixture
def runner():
    return PipelineRunner(
        dataset=None,
        dataconverter=None,
        optimizers=[],
        runtime=None,
    )


@pytest.fixture
def model_mock():
    return Mock(spec=ModelWrapper)


@pytest.fixture
def runtime_mock():
    return Mock(spec=Runtime)


@pytest.fixture
def dataset_mock():
    return Mock(spec=Dataset)


@pytest.fixture
def dataconverter_mock():
    return Mock(spec=DataConverter)


@pytest.fixture
def optimizer_mock():
    return Mock(spec=Optimizer)


def setup_model_mock(model_mock: Mock):
    model_mock.get_path.return_value = "model_path"
    model_mock.save_io_specification.return_value = True
    model_mock.test_inference.return_value = True
    model_mock.get_framework_and_version.return_value = (
        "framework_1",
        "0.0.1",
    )


def setup_protocol_mock(protocol_mock: Mock):
    protocol_mock.upload_model.return_value = True
    protocol_mock.download_statistics.return_value = {}
    protocol_mock.upload_input.return_value = True
    protocol_mock.request_processing.return_value = True
    protocol_mock.download_output.return_value = True
    protocol_mock.upload_optimizers.return_value = True
    protocol_mock.request_optimization.return_value = (
        True,
        bytearray([1, 2, 3, 4]),
    )


def setup_dataset_mock(dataset_mock: Mock):
    dataset_mock.classnames = ["class1", "class2"]
    dataset_mock.get_class_names.return_value = dataset_mock.classnames
    dataset_mock.iter_test.return_value = [([1, 2], 3)]
    dataset_mock._evaluate.return_value = {}


def setup_dataconverter_mock(dataconverter_mock: Mock):
    dataconverter_mock.to_next_block = lambda x: x
    dataconverter_mock.to_previous_block = lambda x: x


class TestPipelineRunnerRun:
    def test_goals_present(self, runner: PipelineRunner):
        with pytest.raises(AssertionError) as e:
            runner.run(run_optimizations=False, run_benchmarks=False)
        assert "If both optimizations and benchmarks" in str(e.value)

    def test_protocol_initialization_called(
        self, runner: PipelineRunner, protocol_mock: Mock
    ):
        protocol_mock.initialize_client.return_value = True
        runner.protocol = protocol_mock
        with pytest.raises(AssertionError) as e:
            runner.run(run_optimizations=True, run_benchmarks=False)
        assert "Model wrapper is required for" in str(e.value)
        assert protocol_mock.initialize_client.called_once()

    def test_model_path_obtained(
        self, runner: PipelineRunner, model_mock: Mock
    ):
        model_mock.get_path.return_value = "model_path"
        model_mock.save_io_specification.return_value = True
        runner.model_wrapper = model_mock
        assert runner.run(run_optimizations=True, run_benchmarks=False) == 0
        assert model_mock.get_path.called_once()
        assert model_mock.save_io_specification.called_once()

        # Local run
        model_mock.reset_mock()
        model_mock.get_path.return_value = "model_mock_path"
        model_mock.save_io_specification.return_value = True
        model_mock.test_inference.return_value = True
        assert runner.run(run_optimizations=True, run_benchmarks=True) == 0
        assert model_mock.test_inference.called_once()

    def test_pipeline_run_runtime(
        self,
        runner: PipelineRunner,
        dataset_mock: Mock,
        dataconverter_mock: Mock,
        model_mock: Mock,
        protocol_mock: Mock,
        runtime_mock: Mock,
        optimizer_mock: Mock,
    ):
        # Local run
        setup_model_mock(model_mock)
        runner.model_wrapper = model_mock
        with open("compiled_model.zip", "wb") as f:
            f.write(bytearray([1, 2, 3, 4]))

        protocol_mock.initialize_client.return_value = True
        runner.protocol = protocol_mock
        assert runner.run(run_optimizations=True, run_benchmarks=True) == 0
        assert model_mock.test_inference.called_once()
        assert runner.run(run_optimizations=False, run_benchmarks=True) == 0

        # Runtime run without dataset
        model_mock.reset_mock()
        model_mock.get_output_formats.return_value = ["format1", "format2"]
        runtime_mock.get_input_formats.return_value = ["format1", "format2"]
        runner.runtime = runtime_mock
        assert runner.run(run_optimizations=True, run_benchmarks=True) == 1

        # Runtime run with dataset
        setup_dataset_mock(dataset_mock)
        runner.dataset = dataset_mock
        setup_protocol_mock(protocol_mock)
        setup_dataconverter_mock(dataconverter_mock)
        runner.dataconverter = dataconverter_mock
        optimizer_mock.location = "target"
        optimizer_mock.compiled_model_path = Path("compiled_model")
        optimizer_mock.get_output_formats.return_value = ["format1", "format2"]
        optimizer_mock.get_input_formats.return_value = ["format1", "format2"]
        optimizer_mock.get_framework_and_version.return_value = (
            "framework_1",
            "0.0.1",
        )
        runner.optimizers = [optimizer_mock]
        assert (
            runner.run(
                run_optimizations=True,
                run_benchmarks=True,
                command=["test command"],
                output=Path("test_measurements.json"),
            )
            == 0
        )
        assert protocol_mock.upload_model.called_once()
        assert dataset_mock.iter_test.called_once()
        assert protocol_mock.download_statistics.called_once()
        assert MeasurementsCollector.measurements.data["command"] == [
            "test command"
        ]
        assert isinstance(
            MeasurementsCollector.measurements.data["build_cfg"], dict
        )
        assert (
            runner.run(
                run_optimizations=False,
                run_benchmarks=True,
                command=["test command"],
                output=Path("test_measurements.json"),
            )
            == 0
        )

        # Run with ONNX conversion
        optimizer_mock.consult_model_type.return_value = "onnx"
        assert (
            runner.run(
                run_optimizations=True,
                run_benchmarks=True,
                command=["test command"],
                output=Path("test_measurements.json"),
            )
            == 0
        )

        runner.model_wrapper = None
        runner.optimizers = []
        assert (
            runner.run(
                run_optimizations=False,
                run_benchmarks=True,
                command=["test command"],
                output=Path("test_measurements.json"),
            )
            == 0
        )

    def test_pipeline_exceptions(
        self,
        runner: PipelineRunner,
        protocol_mock: Mock,
        model_mock: Mock,
        optimizer_mock: Mock,
        dataset_mock: Mock,
        dataconverter_mock: Mock,
        runtime_mock: Mock,
    ):
        setup_model_mock(model_mock)
        runner.model_wrapper = model_mock
        protocol_mock.initialize_client.return_value = True
        setup_protocol_mock(protocol_mock)
        runner.protocol = protocol_mock
        optimizer_mock.location = "target"
        optimizer_mock.compiled_model_path = Path("compiled_model")
        setup_dataset_mock(dataset_mock)
        runner.dataset = dataset_mock
        setup_dataconverter_mock(dataconverter_mock)

        optimizer_mock.get_output_formats.return_value = ["format"]
        optimizer_mock.get_input_formats.return_value = ["format"]
        model_mock.get_output_formats.return_value = ["format"]
        runtime_mock.get_input_formats.return_value = ["format"]
        runtime_mock.get_input_formats.return_value = ["format"]
        runner.runtime = runtime_mock

        # Run with canceled benchmarks
        runner.should_cancel = True
        runner.protocol = None
        assert runner.run(run_optimizations=True, run_benchmarks=True) == 0

        # Run with compilation error
        protocol_mock.request_optimization.return_value = True
        runner.protocol = protocol_mock
        runner.should_cancel = False
        runner.optimizers = [optimizer_mock]
        with pytest.raises(RuntimeError) as ex:
            runner.run(
                run_optimizations=True,
                run_benchmarks=False,
                convert_to_onnx=Path("onnx"),
            )

        assert str(ex.value) == "Model compilation failed"

        # Run without protocol
        runner.protocol = None
        with pytest.raises(RequestFailure) as ex:
            runner._run_client(compiled_model_path=Path("compiled_model"))
        assert str(ex.value) == "Protocol is not provided"

    def test_formats_difference(
        self,
        runner: PipelineRunner,
        protocol_mock: Mock,
        model_mock: Mock,
        optimizer_mock: Mock,
        runtime_mock: Mock,
    ):
        # Run with mismatches in input and output formats
        model_mock.get_output_formats.return_value = ["format1"]
        model_mock.get_path.return_value = "model_path"
        model_mock.save_io_specification.return_value = True
        model_mock.test_inference.return_value = True
        runner.model_wrapper = model_mock
        runtime_mock.get_input_formats.return_value = ["format2"]
        runner.runtime = runtime_mock
        protocol_mock.initialize_client.return_value = True
        protocol_mock.upload_optimizers.return_value = True
        protocol_mock.request_optimization.return_value = (
            True,
            bytearray([1, 2, 3, 4]),
        )
        runner.protocol = protocol_mock
        optimizer_mock.location = "target"
        optimizer_mock.compiled_model_path = Path("compiled_model")
        optimizer_mock.get_output_formats.return_value = ["format3"]
        optimizer_mock.get_input_formats.return_value = ["format4"]
        optimizer_mock.consult_model_type = None
        runner.optimizers = optimizer_mock

        with pytest.raises(ValueError):
            runner.run(run_optimizations=True, run_benchmarks=False)
