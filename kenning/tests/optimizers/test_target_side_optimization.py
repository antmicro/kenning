# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import shutil
import threading
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Literal, Optional, Tuple

import pytest

from kenning.core.optimizer import Optimizer
from kenning.core.protocol import RequestFailure
from kenning.dataconverters.modelwrapper_dataconverter import (
    ModelWrapperDataConverter,
)
from kenning.protocols.network import NetworkProtocol
from kenning.runtimes.tflite import TFLiteRuntime
from kenning.scenarios.inference_server import InferenceServer
from kenning.tests.core.conftest import get_default_dataset_model
from kenning.utils.pipeline_runner import PipelineRunner
from kenning.utils.resource_manager import PathOrURI


class OptimizerMock(Optimizer):
    """
    Optimizer mock that only copies model.
    """

    inputtypes = {"keras": lambda: None}

    outputtypes = ["keras"]

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ):
        shutil.copy(input_model_path, self.compiled_model_path)

    def get_framework_and_version(self) -> Tuple[str, str]:
        return "none", "0"

    def to_json(self) -> Dict[str, Any]:
        ret = super().to_json()
        ret["type"] = (
            "kenning.tests.optimizers.test_target_side_optimization."
            f"{self.__class__.__name__}"
        )
        return ret


class OptimizerFailMock(OptimizerMock):
    """
    Optimizer mock that raises exception.
    """

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ):
        raise ImportError


class TestServerSideOptimization:
    @pytest.mark.xdist_group(name="use_resources")
    def test_local_optimization(self):
        """
        Test local compilation.
        """
        optimizers = [
            OptimizerMock(
                dataset=None,
                compiled_model_path=Path(f"./build/compiled_model_{i}.h5"),
            )
            for i in range(3)
        ]

        runtime_host = TFLiteRuntime(
            model_path=Path("./build/compiled_model.tflite"),
        )

        dataset, model_wrapper = get_default_dataset_model("keras")
        dataconverter = ModelWrapperDataConverter(model_wrapper)

        pipeline_runner = PipelineRunner(
            dataset=dataset,
            dataconverter=dataconverter,
            model_wrapper=model_wrapper,
            optimizers=optimizers,
            runtime=runtime_host,
        )

        model_path = pipeline_runner._handle_optimizations()

        assert model_path.exists()

    @pytest.mark.xdist_group(name="use_socket")
    @pytest.mark.parametrize(
        "optimizers_locations",
        (
            ("host",),
            ("target",),
            ("host", "target"),
            ("target", "host"),
            ("host", "host"),
            ("target", "target"),
            ("host", "target", "host", "target"),
            (
                "host",
                "target",
                "target",
                "target",
                "target",
                "host",
                "host",
                "target",
                "target",
                "host",
            ),
        ),
    )
    def test_target_side_optimization(
        self, optimizers_locations: List[Literal["host", "target"]]
    ):
        """
        Test various target-side compilation scenarios.
        """
        optimizers = [
            OptimizerMock(
                dataset=None,
                compiled_model_path=Path(f"./build/compiled_model_{i}.h5"),
                location=location,
            )
            for i, location in enumerate(optimizers_locations)
        ]

        runtime_target = TFLiteRuntime(
            model_path=Path("./build/compiled_model.tflite"),
        )
        protocol_target = NetworkProtocol("localhost", 12345, 32768)
        inference_server = InferenceServer(
            runtime=runtime_target, protocol=protocol_target
        )

        dataset, model_wrapper = get_default_dataset_model("keras")
        dataconverter = ModelWrapperDataConverter(model_wrapper)
        runtime_host = TFLiteRuntime(
            model_path=Path("./build/compiled_model.tflite"),
        )
        protocol_host = NetworkProtocol("localhost", 12345, 32768)

        pipeline_runner = PipelineRunner(
            dataset=dataset,
            dataconverter=dataconverter,
            model_wrapper=model_wrapper,
            optimizers=optimizers,
            runtime=runtime_host,
            protocol=protocol_host,
        )

        server_thread = threading.Thread(target=inference_server.run)
        try:
            server_thread.start()
            sleep(0.1)

            assert server_thread.is_alive()

            protocol_host.initialize_client()

            model_path = pipeline_runner._handle_optimizations()
            assert model_path.exists()
            assert (
                model_path.read_bytes()
                == model_wrapper.model_path.read_bytes()
            )

        finally:
            inference_server.close()
            server_thread.join()
            assert not server_thread.is_alive()

    @pytest.mark.xdist_group(name="use_socket")
    def test_optimization_fail_when_server_is_not_running(self):
        """
        Test target side optimizations handling when the server is not running.
        """
        optimizers = [
            OptimizerMock(
                dataset=None,
                compiled_model_path=Path(f"./build/compiled_model_{i}.h5"),
                location=location,
            )
            for i, location in enumerate(("target", "host", "target"))
        ]

        dataset, model_wrapper = get_default_dataset_model("keras")
        dataconverter = ModelWrapperDataConverter(model_wrapper)
        runtime = TFLiteRuntime(
            model_path=Path("./build/compiled_model.tflite"),
        )
        protocol = NetworkProtocol("localhost", 12345, 32768)

        pipeline_runner = PipelineRunner(
            dataset=dataset,
            dataconverter=dataconverter,
            model_wrapper=model_wrapper,
            optimizers=optimizers,
            runtime=runtime,
            protocol=protocol,
        )

        with pytest.raises(RequestFailure):
            pipeline_runner._handle_optimizations()

    @pytest.mark.xdist_group(name="use_socket")
    def test_target_side_optimization_compile_fail(self):
        """
        Test various target-side compilation scenarios.
        """
        optimizers = [
            OptimizerMock(
                dataset=None,
                compiled_model_path=Path("./build/compiled_model_0.h5"),
                location="host",
            ),
            OptimizerMock(
                dataset=None,
                compiled_model_path=Path("./build/compiled_model_1.h5"),
                location="target",
            ),
            OptimizerFailMock(
                dataset=None,
                compiled_model_path=Path("./build/compiled_model_0.h5"),
                location="target",
            ),
        ]

        runtime_target = TFLiteRuntime(
            model_path=Path("./build/compiled_model.tflite"),
        )
        protocol_target = NetworkProtocol("localhost", 12345, 32768)
        inference_server = InferenceServer(
            runtime=runtime_target, protocol=protocol_target
        )

        dataset, model_wrapper = get_default_dataset_model("keras")
        dataconverter = ModelWrapperDataConverter(model_wrapper)
        runtime_host = TFLiteRuntime(
            model_path=Path("./build/compiled_model.tflite"),
        )
        protocol_host = NetworkProtocol("localhost", 12345, 32768)

        pipeline_runner = PipelineRunner(
            dataset=dataset,
            dataconverter=dataconverter,
            model_wrapper=model_wrapper,
            optimizers=optimizers,
            runtime=runtime_host,
            protocol=protocol_host,
        )

        server_thread = threading.Thread(target=inference_server.run)
        try:
            server_thread.start()
            sleep(0.1)

            assert server_thread.is_alive()

            protocol_host.initialize_client()

            with pytest.raises(RequestFailure):
                pipeline_runner._handle_optimizations()

        finally:
            inference_server.close()
            server_thread.join()
            assert not server_thread.is_alive()

    @pytest.mark.xdist_group(name="use_socket")
    def test_optimization_when_protocol_is_not_specified(self):
        """
        Test target side optimizations handling when protocol is not specified.
        """
        optimizers = [
            OptimizerMock(
                dataset=None,
                compiled_model_path=Path(f"./build/compiled_model_{i}.h5"),
                location=location,
            )
            for i, location in enumerate(("target", "host", "target"))
        ]

        dataset, model_wrapper = get_default_dataset_model("keras")
        dataconverter = ModelWrapperDataConverter(model_wrapper)
        runtime = TFLiteRuntime(
            model_path=Path("./build/compiled_model.tflite"),
        )

        pipeline_runner = PipelineRunner(
            dataset=dataset,
            dataconverter=dataconverter,
            model_wrapper=model_wrapper,
            optimizers=optimizers,
            runtime=runtime,
            protocol=None,
        )

        model_path = pipeline_runner._handle_optimizations()

        assert model_path.exists()
        assert model_path.read_bytes() == model_wrapper.model_path.read_bytes()

    @pytest.mark.xdist_group(name="use_socket")
    @pytest.mark.parametrize("max_optimizers", (1, 2, 4, 8))
    def test_limit_target_side_optimization(self, max_optimizers: int):
        """
        Test various target-side compilation scenarios.
        """
        optimizers = [
            OptimizerMock(
                dataset=None,
                compiled_model_path=Path(f"./build/compiled_model_{i}.h5"),
                location=location,
            )
            for i, location in enumerate(("target",) * 6)
        ]

        max_loaded_optimizers = -1

        prev_callback = InferenceServer._optimizers_callback

        def optimizers_callback_mock(self, input_data: bytes) -> bool:
            nonlocal max_loaded_optimizers
            ret = prev_callback(self, input_data)
            max_loaded_optimizers = max(
                max_loaded_optimizers, len(self.optimizers)
            )
            return ret

        InferenceServer._optimizers_callback = optimizers_callback_mock

        runtime_target = TFLiteRuntime(
            model_path=Path("./build/compiled_model.tflite"),
        )
        protocol_target = NetworkProtocol("localhost", 12345, 32768)
        inference_server = InferenceServer(
            runtime=runtime_target, protocol=protocol_target
        )

        dataset, model_wrapper = get_default_dataset_model("keras")
        dataconverter = ModelWrapperDataConverter(model_wrapper)
        runtime_host = TFLiteRuntime(
            model_path=Path("./build/compiled_model.tflite"),
        )
        protocol_host = NetworkProtocol("localhost", 12345, 32768)

        pipeline_runner = PipelineRunner(
            dataset=dataset,
            dataconverter=dataconverter,
            model_wrapper=model_wrapper,
            optimizers=optimizers,
            runtime=runtime_host,
            protocol=protocol_host,
        )

        server_thread = threading.Thread(target=inference_server.run)
        try:
            server_thread.start()
            sleep(0.1)

            assert server_thread.is_alive()

            protocol_host.initialize_client()

            pipeline_runner._handle_optimizations(
                max_target_side_optimizers=max_optimizers
            )
            assert max_loaded_optimizers <= max_optimizers

        finally:
            inference_server.close()
            server_thread.join()
            assert not server_thread.is_alive()
