# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from kenning.core.protocol import Protocol
from kenning.core.model import ModelWrapper
from kenning.core.runtime import Runtime
from kenning.core.dataset import Dataset
from kenning.utils.pipeline_runner import PipelineRunner
import pytest
from unittest.mock import Mock


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


class TestPipelineRunnerRun:
    def test_goals_present(self, runner: PipelineRunner):
        with pytest.raises(AssertionError) as e:
            runner.run(run_optimizations=False, run_benchmarks=False)
        assert 'If both optimizations and benchmarks' in str(e.value)

    def test_protocol_initialization_called(self,
                                            runner: PipelineRunner,
                                            protocol_mock: Mock):
        protocol_mock.initialize_client.return_value = True
        runner.protocol = protocol_mock
        with pytest.raises(AssertionError) as e:
            runner.run(run_optimizations=True, run_benchmarks=False)
        assert 'Model wrapper is required for' in str(e.value)
        assert protocol_mock.initialize_client.called_once()

    def test_model_path_obtained(self,
                                 runner: PipelineRunner,
                                 model_mock: Mock):
        model_mock.get_path.return_value = 'model_path'
        model_mock.save_io_specification.return_value = True
        runner.model_wrapper = model_mock
        assert runner.run(run_optimizations=True, run_benchmarks=False) == 0
        assert model_mock.get_path.called_once()
        assert model_mock.save_io_specification.called_once()

        # Local run
        model_mock.reset_mock()
        model_mock.get_path.return_value = 'model_mock_path'
        model_mock.save_io_specification.return_value = True
        model_mock.test_inference.return_value = True
        assert runner.run(run_optimizations=True, run_benchmarks=True) == 0
        assert model_mock.test_inference.called_once()

    def test_pipeline_run_runtime(
            self,
            runner: PipelineRunner,
            dataset_mock: Mock,
            model_mock: Mock,
            protocol_mock: Mock,
            runtime_mock: Mock,
            ):
        # Local run
        model_mock.get_path.return_value = 'model_path'
        model_mock.save_io_specification.return_value = True
        model_mock.test_inference.return_value = True
        protocol_mock.initialize_client.return_value = True
        runner.model_wrapper = model_mock
        runner.protocol = protocol_mock
        assert runner.run(run_optimizations=True, run_benchmarks=True) == 0
        assert model_mock.test_inference.called_once()

        # Runtime run without dataset
        model_mock.reset_mock()
        model_mock.get_output_formats.return_value = ['format1', 'format2']
        runtime_mock.get_input_formats.return_value = ['format1', 'format2']
        runner.runtime = runtime_mock
        assert runner.run(run_optimizations=True, run_benchmarks=True) == 1

        # Runtime run with dataset
        runner.dataset = dataset_mock
        protocol_mock.upload_model.return_value = True
        protocol_mock.download_statistics.return_value = {}
        dataset_mock.iter_test.return_value = []
        assert runner.run(run_optimizations=True, run_benchmarks=True) == 0
        assert protocol_mock.upload_model.called_once()
        assert dataset_mock.iter_test.called_once()
        assert protocol_mock.download_statistics.called_once()
