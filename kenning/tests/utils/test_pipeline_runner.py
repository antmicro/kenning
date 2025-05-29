# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pytest

from kenning.core.dataconverter import DataConverter
from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import OptimizedModelSizeError, Optimizer
from kenning.core.platform import Platform
from kenning.core.protocol import Protocol
from kenning.core.runtime import Runtime
from kenning.core.runtimebuilder import RuntimeBuilder
from kenning.platforms.zephyr import ZephyrPlatform as _ZephyrPlatform
from kenning.runtimebuilders.zephyr import (
    ZephyrRuntimeBuilder as _ZephyrRuntimeBuilder,
)
from kenning.utils.pipeline_runner import (
    PipelineRunner,
    PipelineRunnerInvalidConfigError,
)


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
    mock = Mock(spec=ModelWrapper)
    mock.get_path.return_value = "model_path"
    mock.get_output_formats.return_value = ["format1", "format2"]
    mock.save_io_specification.return_value = True
    mock.test_inference.return_value = True
    mock.get_framework_and_version.return_value = (
        "framework_1",
        "0.0.1",
    )
    return mock


@pytest.fixture
def protocol_mock():
    mock = Mock(spec=Protocol)
    mock.initialize_client.return_value = True
    mock.upload_io_specification.return_value = True
    mock.upload_model.return_value = True
    mock.download_statistics.return_value = {}
    mock.upload_input.return_value = True
    mock.request_processing.return_value = True
    mock.download_output.return_value = True
    mock.upload_optimizers.return_value = True
    mock.upload_runtime.return_value = True
    mock.request_optimization.return_value = (
        True,
        bytearray([1, 2, 3, 4]),
    )
    return mock


@pytest.fixture
def runtime_mock():
    return Mock(spec=Runtime)


@pytest.fixture
def dataset_mock():
    mock = Mock(spec=Dataset)
    mock.classnames = ["class1", "class2"]
    mock.get_class_names.return_value = mock.classnames
    mock.iter_test.return_value = [([1, 2], 3)]
    mock._evaluate.return_value = {}
    return mock


@pytest.fixture
def dataconverter_mock():
    mock = Mock(spec=DataConverter)
    mock.to_next_block = lambda x: x
    mock.to_previous_block = lambda x: x
    return mock


@pytest.fixture
def optimizer_mock():
    mock = Mock(spec=Optimizer)
    mock.get_framework_and_version.return_value = (
        "framework_1",
        "0.0.1",
    )
    mock.get_optimized_model_size.side_effect = OptimizedModelSizeError()
    return mock


@pytest.fixture
def platform_mock():
    return Mock(spec=Platform)


@pytest.fixture
def runtime_builder_mock():
    return Mock(spec=RuntimeBuilder)


TEST_EXAMPLE_VALID_CFG = {
    "model_wrapper": {
        "type": "MagicWandModelWrapper",
        "parameters": {"model_path": "magic_wand.h5"},
    },
    "dataset": {
        "type": "MagicWandDataset",
        "parameters": {
            "dataset_root": "build/MagicWandDataset",
            "download_dataset": False,
        },
    },
    "optimizers": [
        {
            "type": "TFLiteCompiler",
            "parameters": {
                "target": "default",
                "compiled_model_path": "build/fp32.tflite",
                "inference_input_type": "float32",
                "inference_output_type": "float32",
            },
        }
    ],
    "runtime": {
        "type": "TFLiteRuntime",
        "parameters": {"save_model_path": "build/fp32.tflite"},
    },
}
TEST_EXAMPLE_INVALID_CFG = {
    "model_wrapper": {
        "type": "MagicWandModelWrapper",
        "parameters": {"model_path": "magic_wand.h5"},
    },
    "dataset": {
        "type": "MagicWandDataset",
        "parameters": {
            "dataset_root": "build/MagicWandDataset",
            "download_dataset": False,
        },
    },
    "optimizers": [
        {
            "type": "IREECompiler",
            "parameters": {
                "compiled_model_path": "build/tflite-magic-wand.vmfb",
                "backend": "llvm-cpu",
            },
        },
        {
            "type": "TFLiteCompiler",
            "parameters": {
                "target": "default",
                "compiled_model_path": "build/fp32.tflite",
                "inference_input_type": "float32",
                "inference_output_type": "float32",
            },
        },
    ],
    "runtime": {
        "type": "TFLiteRuntime",
        "parameters": {"save_model_path": "./build/fp32.tflite"},
    },
}


class TestPipelineRunnerRun:
    def test_init_no_dataconverter_and_modelwrapper(self):
        with pytest.raises(PipelineRunnerInvalidConfigError) as e:
            PipelineRunner(dataset=None, dataconverter=None, optimizers=[])

        assert "Provide either dataconverter or model_wrapper" in str(e.value)

    def test_from_json_cfg(self):
        PipelineRunner.from_json_cfg(TEST_EXAMPLE_VALID_CFG)

    def test_from_json_cfg_assert_integrity(self):
        PipelineRunner.from_json_cfg(
            TEST_EXAMPLE_VALID_CFG, assert_integrity=True
        )

    def test_from_json_cfg_assert_integrity_error(self):
        with pytest.raises(ValueError):
            PipelineRunner.from_json_cfg(
                TEST_EXAMPLE_INVALID_CFG, assert_integrity=True
            )

    def test_serialize(self):
        def check_block(expected_block, block):
            assert block["type"].rsplit(".", 1)[-1] == expected_block["type"]
            for param, value in expected_block["parameters"].items():
                assert block["parameters"][param] == value

        runner = PipelineRunner.from_json_cfg(TEST_EXAMPLE_VALID_CFG)

        runner_serialized = runner.serialize()

        assert isinstance(runner_serialized, dict)
        for block, block_cfg in TEST_EXAMPLE_VALID_CFG.items():
            if isinstance(block_cfg, list):
                for cfg, set_cfg in zip(block_cfg, runner_serialized[block]):
                    check_block(cfg, set_cfg)
            else:
                check_block(block_cfg, runner_serialized[block])

    def test_add_scenario_cfg_to_measurements(self):
        def check_block(expected_block, block):
            assert block["type"].rsplit(".", 1)[-1] == expected_block["type"]
            for param, value in expected_block["parameters"].items():
                assert block["parameters"][param] == value

        runner = PipelineRunner.from_json_cfg(TEST_EXAMPLE_VALID_CFG)

        runner.add_scenario_configuration_to_measurements("cmd")

    def test_run_targets_present(self, dataconverter_mock: Mock):
        runner = PipelineRunner(
            dataset=None, dataconverter=dataconverter_mock, optimizers=[]
        )

        with pytest.raises(PipelineRunnerInvalidConfigError) as e:
            runner.run(run_optimizations=False, run_benchmarks=False)

        assert "If both optimizations and benchmarks" in str(e.value)

    def test_run_model_required_for_optimization(
        self, dataconverter_mock: Mock
    ):
        runner = PipelineRunner(
            dataset=None,
            dataconverter=dataconverter_mock,
            optimizers=[],
        )

        with pytest.raises(PipelineRunnerInvalidConfigError) as e:
            runner.run(run_optimizations=True, run_benchmarks=False)

        assert "Model wrapper is required for" in str(e.value)

    def test_run_model_path_obtained(self, model_mock: Mock):
        model_mock.get_path.return_value = "model_path"
        model_mock.save_io_specification.return_value = True
        runner = PipelineRunner(dataset=None, model_wrapper=model_mock)

        runner.run(run_optimizations=False, run_benchmarks=True)

        model_mock.get_path.assert_called_once()
        model_mock.save_io_specification.assert_called_once()

    def test_run_local_model_test_inference(self, model_mock: Mock):
        model_mock.get_path.return_value = "model_mock_path"
        model_mock.save_io_specification.return_value = True
        model_mock.test_inference.return_value = True
        runner = PipelineRunner(dataset=None, model_wrapper=model_mock)

        runner.run(run_optimizations=True, run_benchmarks=True)

        model_mock.test_inference.assert_called_once()

    def test_run_local_runtime(
        self, dataset_mock: Mock, model_mock: Mock, runtime_mock: Mock
    ):
        dataset_mock.iter_test.return_value = [[0, 0]]
        model_mock.get_path.return_value = "model_mock_path"
        model_mock.save_io_specification.return_value = True
        runner = PipelineRunner(
            dataset=dataset_mock,
            model_wrapper=model_mock,
            runtime=runtime_mock,
        )

        runner.run(run_optimizations=True, run_benchmarks=True)

        runtime_mock.inference_session_start.assert_called_once()
        runtime_mock.load_input.assert_called_once()
        runtime_mock._run.assert_called_once()
        runtime_mock.extract_output.assert_called_once()
        runtime_mock.inference_session_end.assert_called_once()

    def test_run_local_output(
        self,
        tmpfolder: Path,
        dataset_mock: Mock,
        model_mock: Mock,
        runtime_mock: Mock,
    ):
        dataset_mock.iter_test.return_value = [[0, 0]]
        model_path = tmpfolder / "model"
        model_path.touch()
        model_mock.get_path.return_value = model_path
        model_mock.save_io_specification.return_value = True
        output_path = tmpfolder / "output.json"
        runner = PipelineRunner(
            dataset=dataset_mock,
            model_wrapper=model_mock,
            runtime=runtime_mock,
        )

        runner.run(
            output=output_path, run_optimizations=True, run_benchmarks=True
        )

        assert output_path.exists()

    def test_run_remote(
        self,
        dataset_mock: Mock,
        model_mock: Mock,
        runtime_mock: Mock,
        protocol_mock: Mock,
        platform_mock: Mock,
    ):
        dataset_mock.iter_test.return_value = [[0, 0]]
        model_mock.get_path.return_value = "model_mock_path"
        model_mock.save_io_specification.return_value = True
        platform_mock.needs_protocol = True
        runner = PipelineRunner(
            dataset=dataset_mock,
            model_wrapper=model_mock,
            runtime=runtime_mock,
            protocol=protocol_mock,
            platform=platform_mock,
        )

        runner.run(run_optimizations=True, run_benchmarks=True)

        protocol_mock.initialize_client.assert_called_once()
        protocol_mock.upload_io_specification.assert_called_once()
        protocol_mock.upload_model.assert_called_once()
        protocol_mock.upload_input.assert_called_once()
        protocol_mock.request_processing.assert_called_once()
        protocol_mock.download_output.assert_called_once()
        protocol_mock.disconnect.assert_called_once()

        assert protocol_mock.download_statistics.call_args_list == [
            ({"final": False},),
            ({"final": True},),
        ]

    def test_run_remote_no_runtime(
        self,
        dataset_mock: Mock,
        model_mock: Mock,
        protocol_mock: Mock,
        platform_mock: Mock,
    ):
        dataset_mock.iter_test.return_value = [[0, 0]]
        model_mock.get_path.return_value = "model_mock_path"
        model_mock.save_io_specification.return_value = True
        platform_mock.needs_protocol = True
        runner = PipelineRunner(
            platform=platform_mock,
            dataset=dataset_mock,
            model_wrapper=model_mock,
            protocol=protocol_mock,
        )

        runner.run(run_optimizations=True, run_benchmarks=True)

    def test_run_local_optimizations(
        self,
        dataset_mock: Mock,
        model_mock: Mock,
        protocol_mock: Mock,
        optimizer_mock: Mock,
        runtime_mock: Mock,
    ):
        model_mock.get_output_formats.return_value = ["format1"]
        runtime_mock.get_input_formats.return_value = ["format2"]
        protocol_mock.request_optimization.return_value = (
            True,
            bytearray([1, 2, 3, 4]),
        )
        optimizer_mock.location = "host"
        optimizer_mock.compiled_model_path = Path("compiled_model")
        optimizer_mock.get_input_formats.return_value = ["format1"]
        optimizer_mock.get_output_formats.return_value = ["format2"]
        optimizer_mock.consult_model_type = (
            lambda prev_block, force_onnx: Optimizer.consult_model_type(
                optimizer_mock, prev_block, force_onnx
            )
        )

        runner = PipelineRunner(
            dataset=dataset_mock,
            model_wrapper=model_mock,
            runtime=runtime_mock,
            protocol=protocol_mock,
            optimizers=[optimizer_mock],
        )

        runner.run(run_optimizations=True, run_benchmarks=False)

        optimizer_mock.compile.assert_called_once()

    def test_run_local_optimizations_onnx(
        self,
        dataset_mock: Mock,
        model_mock: Mock,
        protocol_mock: Mock,
        optimizer_mock: Mock,
        runtime_mock: Mock,
    ):
        model_mock.get_output_formats.return_value = ["onnx"]
        runtime_mock.get_input_formats.return_value = ["onnx"]
        protocol_mock.request_optimization.return_value = (
            True,
            bytearray([1, 2, 3, 4]),
        )
        optimizer_mock.location = "host"
        optimizer_mock.compiled_model_path = Path("compiled_model")
        optimizer_mock.get_input_formats.return_value = ["onnx"]
        optimizer_mock.get_output_formats.return_value = ["onnx"]
        optimizer_mock.consult_model_type = (
            lambda prev_block, force_onnx: Optimizer.consult_model_type(
                optimizer_mock, prev_block, force_onnx
            )
        )

        runner = PipelineRunner(
            dataset=dataset_mock,
            model_wrapper=model_mock,
            runtime=runtime_mock,
            protocol=protocol_mock,
            optimizers=[optimizer_mock],
        )

        runner.run(
            run_optimizations=True, run_benchmarks=False, convert_to_onnx=True
        )

        optimizer_mock.compile.assert_called_once()

    def test_run_remote_optimizations(
        self,
        dataset_mock: Mock,
        model_mock: Mock,
        protocol_mock: Mock,
        optimizer_mock: Mock,
        runtime_mock: Mock,
    ):
        model_mock.get_output_formats.return_value = ["format1"]
        runtime_mock.get_input_formats.return_value = ["format2"]
        protocol_mock.request_optimization.return_value = (
            True,
            bytearray([1, 2, 3, 4]),
        )
        optimizer_mock.location = "target"
        optimizer_mock.compiled_model_path = Path("compiled_model")
        optimizer_mock.get_input_formats.return_value = ["format1"]
        optimizer_mock.get_output_formats.return_value = ["format2"]
        optimizer_mock.consult_model_type = (
            lambda prev_block, force_onnx: Optimizer.consult_model_type(
                optimizer_mock, prev_block, force_onnx
            )
        )

        runner = PipelineRunner(
            dataset=dataset_mock,
            model_wrapper=model_mock,
            runtime=runtime_mock,
            protocol=protocol_mock,
            optimizers=[optimizer_mock],
        )

        runner.run(run_optimizations=True, run_benchmarks=False)

        protocol_mock.request_optimization.assert_called_once()

    def test_run_remote_no_optimizations(
        self,
        tmpfolder: Path,
        dataset_mock: Mock,
        model_mock: Mock,
        protocol_mock: Mock,
        optimizer_mock: Mock,
        runtime_mock: Mock,
    ):
        model_mock.get_output_formats.return_value = ["format1"]
        runtime_mock.get_input_formats.return_value = ["format2"]
        model_path = tmpfolder / "model"
        model_path.write_bytes(b"\x01\x02\x03\x04")
        optimizer_mock.compiled_model_path = model_path
        optimizer_mock.get_input_formats.return_value = ["format1"]
        optimizer_mock.get_output_formats.return_value = ["format2"]
        optimizer_mock.consult_model_type = (
            lambda prev_block, force_onnx: Optimizer.consult_model_type(
                optimizer_mock, prev_block, force_onnx
            )
        )
        runner = PipelineRunner(
            dataset=dataset_mock,
            model_wrapper=model_mock,
            runtime=runtime_mock,
            protocol=protocol_mock,
            optimizers=[optimizer_mock],
        )
        output_path = tmpfolder / "output.json"

        runner.run(output_path, run_optimizations=False, run_benchmarks=True)

        assert json.loads(output_path.read_text())["compiled_model_size"] == 4

    def test_run_remote_no_optimizations_no_optimizers(
        self,
        tmpfolder: Path,
        dataset_mock: Mock,
        model_mock: Mock,
        protocol_mock: Mock,
        runtime_mock: Mock,
    ):
        model_path = tmpfolder / "model"
        model_path.write_bytes(b"\x01\x02\x03\x04\x05")
        model_mock.get_path.return_value = model_path
        runner = PipelineRunner(
            dataset=dataset_mock,
            model_wrapper=model_mock,
            runtime=runtime_mock,
            protocol=protocol_mock,
        )
        output_path = tmpfolder / "output.json"

        runner.run(output_path, run_optimizations=False, run_benchmarks=True)

        assert json.loads(output_path.read_text())["compiled_model_size"] == 5

    def test_run_remote_no_optimizations_no_optimizers_no_model(
        self,
        tmpfolder: Path,
        dataset_mock: Mock,
        dataconverter_mock: Mock,
        protocol_mock: Mock,
        runtime_mock: Mock,
    ):
        runner = PipelineRunner(
            dataset=dataset_mock,
            runtime=runtime_mock,
            dataconverter=dataconverter_mock,
            protocol=protocol_mock,
        )
        output_path = tmpfolder / "output.json"

        runner.run(output_path, run_optimizations=False, run_benchmarks=True)

        assert "compiled_model_size" not in json.loads(output_path.read_text())

    def test_run_remote_optimizations_no_protocol(
        self,
        dataset_mock: Mock,
        model_mock: Mock,
        optimizer_mock: Mock,
        runtime_mock: Mock,
    ):
        model_mock.get_output_formats.return_value = ["format1"]
        runtime_mock.get_input_formats.return_value = ["format2"]
        optimizer_mock.location = "target"
        optimizer_mock.get_input_formats.return_value = ["format1"]
        optimizer_mock.get_output_formats.return_value = ["format2"]
        optimizer_mock.consult_model_type = (
            lambda prev_block, force_onnx: Optimizer.consult_model_type(
                optimizer_mock, prev_block, force_onnx
            )
        )

        runner = PipelineRunner(
            dataset=dataset_mock,
            model_wrapper=model_mock,
            runtime=runtime_mock,
            optimizers=[optimizer_mock],
        )

        with pytest.raises(PipelineRunnerInvalidConfigError):
            runner.run(run_optimizations=True, run_benchmarks=False)

    def test_run_remote_optimizations_on_board(
        self,
        dataset_mock: Mock,
        model_mock: Mock,
        protocol_mock: Mock,
        optimizer_mock: Mock,
        runtime_mock: Mock,
    ):
        class ZephyrPlatform(Mock):
            needs_protocol = True

        platform_mock = ZephyrPlatform(spec=_ZephyrPlatform)
        model_mock.get_output_formats.return_value = ["format1"]
        runtime_mock.get_input_formats.return_value = ["format2"]
        optimizer_mock.location = "target"
        optimizer_mock.get_input_formats.return_value = ["format1"]
        optimizer_mock.get_output_formats.return_value = ["format2"]
        optimizer_mock.consult_model_type = (
            lambda prev_block, force_onnx: Optimizer.consult_model_type(
                optimizer_mock, prev_block, force_onnx
            )
        )

        runner = PipelineRunner(
            platform=platform_mock,
            dataset=dataset_mock,
            model_wrapper=model_mock,
            runtime=runtime_mock,
            protocol=protocol_mock,
            optimizers=[optimizer_mock],
        )

        with pytest.raises(PipelineRunnerInvalidConfigError):
            runner.run(run_optimizations=True, run_benchmarks=False)

    def test_run_formats_difference(
        self,
        dataset_mock: Mock,
        protocol_mock: Mock,
        model_mock: Mock,
        optimizer_mock: Mock,
        runtime_mock: Mock,
    ):
        # Run with mismatches in input and output formats
        model_mock.get_output_formats.return_value = ["format1"]
        runtime_mock.get_input_formats.return_value = ["format2"]
        optimizer_mock.location = "target"
        optimizer_mock.get_input_formats.return_value = ["format3"]
        optimizer_mock.get_output_formats.return_value = ["format4"]
        optimizer_mock.consult_model_type = (
            lambda prev_block, force_onnx: Optimizer.consult_model_type(
                optimizer_mock, prev_block, force_onnx
            )
        )

        runner = PipelineRunner(
            dataset=dataset_mock,
            model_wrapper=model_mock,
            runtime=runtime_mock,
            protocol=protocol_mock,
            optimizers=[optimizer_mock],
        )

        with pytest.raises(ValueError):
            runner.run(run_optimizations=True, run_benchmarks=False)

    def test_run_runtime_builder(
        self,
        dataset_mock: Mock,
        model_mock: Mock,
        runtime_mock: Mock,
        runtime_builder_mock: Mock,
    ):
        dataset_mock.iter_test.return_value = [[0, 0]]
        model_mock.get_path.return_value = "model_mock_path"
        model_mock.save_io_specification.return_value = True
        runtime_builder_mock.use_llext = False
        runner = PipelineRunner(
            dataset=dataset_mock,
            model_wrapper=model_mock,
            runtime=runtime_mock,
            runtime_builder=runtime_builder_mock,
        )

        runner.run(run_optimizations=True, run_benchmarks=True)

        runtime_builder_mock.build.assert_called_once()

    def test_run_zephyr_runtime_builder(
        self,
        dataset_mock: Mock,
        model_mock: Mock,
        protocol_mock: Mock,
        runtime_mock: Mock,
    ):
        class ZephyrPlatform(Mock):
            needs_protocol = True

        class ZephyrRuntimeBuilder(Mock):
            pass

        platform_mock = ZephyrPlatform(spec=_ZephyrPlatform)
        platform_mock.zephyr_build_path = None
        platform_mock.name = "test_board"
        runtime_builder_mock = ZephyrRuntimeBuilder(spec=_ZephyrRuntimeBuilder)
        runtime_builder_mock.output_path = Path("test/path")
        dataset_mock.iter_test.return_value = [[0, 0]]
        model_mock.get_path.return_value = "model_mock_path"
        model_mock.save_io_specification.return_value = True
        runtime_builder_mock.use_llext = False
        runner = PipelineRunner(
            platform=platform_mock,
            dataset=dataset_mock,
            model_wrapper=model_mock,
            runtime=runtime_mock,
            protocol=protocol_mock,
            runtime_builder=runtime_builder_mock,
        )

        runner.run(run_optimizations=True, run_benchmarks=True)

        runtime_builder_mock.build.assert_called_once()

    def test_run_zephyr_runtime_builder_llext(
        self,
        dataset_mock: Mock,
        model_mock: Mock,
        protocol_mock: Mock,
        runtime_mock: Mock,
    ):
        class ZephyrPlatform(Mock):
            needs_protocol = True

        class ZephyrRuntimeBuilder(Mock):
            pass

        platform_mock = ZephyrPlatform(spec=_ZephyrPlatform)
        platform_mock.zephyr_build_path = None
        platform_mock.name = "test_board"
        runtime_builder_mock = ZephyrRuntimeBuilder(spec=_ZephyrRuntimeBuilder)
        dataset_mock.iter_test.return_value = [[0, 0]]
        model_mock.get_path.return_value = "model_mock_path"
        model_mock.save_io_specification.return_value = True
        runtime_builder_mock.use_llext = True
        with TemporaryDirectory() as llext_dir:
            llext_path = Path(llext_dir) / "runtime.llext"
            with llext_path.open("wb") as f:
                f.write(b"12345")
            runtime_builder_mock.output_path = Path(llext_dir)
            runner = PipelineRunner(
                platform=platform_mock,
                dataset=dataset_mock,
                model_wrapper=model_mock,
                runtime=runtime_mock,
                protocol=protocol_mock,
                runtime_builder=runtime_builder_mock,
            )

            runner.run(run_optimizations=True, run_benchmarks=True)

            runtime_builder_mock.build.assert_called_once()
            protocol_mock.upload_runtime.assert_called_once()
