# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Tuple,
)

import pytest

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.core.protocol import RequestFailure
from kenning.dataconverters.modelwrapper_dataconverter import (
    ModelWrapperDataConverter,
)
from kenning.protocols.network import NetworkProtocol
from kenning.runtimes.tflite import TFLiteRuntime
from kenning.scenarios.inference_server import InferenceServer
from kenning.tests.core.conftest import DatasetModelRegistry
from kenning.utils.pipeline_runner import PipelineRunner
from kenning.utils.resource_manager import PathOrURI


@contextmanager
def prepare_objects(
    framework: str
) -> Generator[Tuple[Dataset, ModelWrapper], None, None]:
    """
    Context manager to prepare mock dataset and model wrapper for tests.

    Parameters
    ----------
    framework : str
        Name of the framework.

    Yields
    ------
    Generator[Tuple[Dataset, ModelWrapper], None, None]
        Tuple with dataset mock and model wrapper in a temporary location.
    """
    dataset, model_wrapper, assets_id = DatasetModelRegistry.get(framework)
    try:
        yield dataset, model_wrapper
    finally:
        DatasetModelRegistry.remove(assets_id)


TIMEOUT: int = 30


@contextmanager
def running_inference_server(
    inference_server: InferenceServer,
    protocol_client: NetworkProtocol,
) -> Generator[None, None, None]:
    """

    Parameters
    ----------
        inference_server: InferenceServer
            Inference server to run.

        protocol_client: NetworkProtocol
            Client side connection, which should be initialized after server
            start.

    Yields
    ------
    Generator[None, None, None]
        Successful connection readiness sign.
    """
    server_thread = threading.Thread(
        target=inference_server.run,
        args=(),
    )
    server_thread.start()
    deadline: float = time.time() + TIMEOUT

    try:
        while not inference_server.serving_event.wait(0.1):
            assert (
                server_thread.is_alive()
            ), "Inference server stopped before serving start."
            assert (
                time.time() < deadline
            ), f"Server failed to initialize in {TIMEOUT} seconds."

        assert protocol_client.initialize_client(), "Client failed to connect"
        yield
    finally:
        protocol_client.disconnect()
        inference_server.close()
        server_thread.join(TIMEOUT)
        assert not server_thread.is_alive(), "Server thread did not finish."


class OptimizerMock(Optimizer):
    """
    Optimizer mock that only copies model.
    """

    inputtypes = ["keras"]
    outputtypes = ["keras"]

    mock_counter = 0

    @classmethod
    def get_new_mock(cls, *args, **kwargs):
        # Problem: When trying to use multiple optimizers at once, with the
        # same class name, the class_loader goes crazy. Therefore we create a
        # new class for every instance, adding a number to the name.
        name = f"{cls.__name__}_{str(cls.mock_counter)}"
        cls.mock_counter += 1
        mock_optimizer_class = type(name, (cls,), {})
        # For this to actually work, we need to add the class to the module,
        # otherwise class_loader won't find it.
        import kenning.tests.optimizers.test_target_side_optimization

        setattr(
            kenning.tests.optimizers.test_target_side_optimization,
            name,
            mock_optimizer_class,
        )
        return mock_optimizer_class(*args, **kwargs)

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
        **kwargs: Dict,
    ):
        shutil.copy(input_model_path, self.compiled_model_path)

    @classmethod
    def get_framework(cls) -> str:
        return "none"

    @classmethod
    def get_framework_version(cls) -> str:
        return "0"

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
        **kwargs: Dict,
    ):
        raise ImportError


class TestServerSideOptimization:
    def test_local_optimization(self):
        """
        Test local compilation.
        """
        optimizers = [
            OptimizerMock.get_new_mock(
                dataset=None,
                compiled_model_path=Path(f"./build/compiled_model_{i}.h5"),
            )
            for i in range(3)
        ]

        runtime_host = TFLiteRuntime(
            model_path=Path("./build/compiled_model.tflite"),
        )

        with prepare_objects("keras") as (dataset, model_wrapper):
            dataconverter = ModelWrapperDataConverter(model_wrapper)

            pipeline_runner = PipelineRunner(
                dataset=dataset,
                dataconverter=dataconverter,
                model_wrapper=model_wrapper,
                optimizers=optimizers,
                runtime=runtime_host,
            )

            model_path = pipeline_runner._handle_optimizations()

            assert model_path and model_path.exists()

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
            OptimizerMock.get_new_mock(
                dataset=None,
                compiled_model_path=Path(f"./build/compiled_model_{i}.h5"),
                location=location,
            )
            for i, location in enumerate(optimizers_locations)
        ]
        runtime_target = TFLiteRuntime(
            model_path=Path("./build/compiled_model.tflite"),
        )
        protocol_target = NetworkProtocol(
            "localhost", port=12345, timeout=int(TIMEOUT), packet_size=32768
        )
        inference_server = InferenceServer(
            runtime=runtime_target, protocol=protocol_target
        )

        with prepare_objects("keras") as (dataset, model_wrapper):
            dataconverter = ModelWrapperDataConverter(model_wrapper)
            runtime_host = TFLiteRuntime(
                model_path=Path("./build/compiled_model.tflite"),
            )
            protocol_host = NetworkProtocol(
                "localhost",
                port=12345,
                timeout=int(TIMEOUT),
                packet_size=32768,
            )

            pipeline_runner = PipelineRunner(
                dataset=dataset,
                dataconverter=dataconverter,
                model_wrapper=model_wrapper,
                optimizers=optimizers,
                runtime=runtime_host,
                protocol=protocol_host,
            )

            with running_inference_server(inference_server, protocol_host):
                model_path = pipeline_runner._handle_optimizations()
                assert model_path and model_path.exists()
                assert (
                    model_path.read_bytes()
                    == model_wrapper.model_path.read_bytes()
                )

    @pytest.mark.xdist_group(name="use_socket")
    def test_target_side_optimization_compile_fail(self):
        """
        Test various target-side compilation scenarios.
        """
        optimizers = [
            OptimizerMock.get_new_mock(
                dataset=None,
                compiled_model_path=Path("./build/compiled_model_0.h5"),
                location="host",
            ),
            OptimizerMock.get_new_mock(
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
        protocol_target = NetworkProtocol(
            "localhost", 12345, timeout=TIMEOUT, packet_size=32768
        )
        inference_server = InferenceServer(
            runtime=runtime_target, protocol=protocol_target
        )

        with prepare_objects("keras") as (dataset, model_wrapper):
            dataconverter = ModelWrapperDataConverter(model_wrapper)
            runtime_host = TFLiteRuntime(
                model_path=Path("./build/compiled_model.tflite"),
            )
            protocol_host = NetworkProtocol(
                "localhost", 12345, timeout=TIMEOUT, packet_size=32768
            )

            pipeline_runner = PipelineRunner(
                dataset=dataset,
                dataconverter=dataconverter,
                model_wrapper=model_wrapper,
                optimizers=optimizers,
                runtime=runtime_host,
                protocol=protocol_host,
            )

            with running_inference_server(
                inference_server, protocol_host
            ), pytest.raises(RequestFailure):
                pipeline_runner._handle_optimizations()

    @pytest.mark.xdist_group(name="use_socket")
    def test_optimization_when_protocol_is_not_specified(self):
        """
        Test target side optimizations handling when protocol is not specified.
        """
        optimizers = [
            OptimizerMock.get_new_mock(
                dataset=None,
                compiled_model_path=Path(f"./build/compiled_model_{i}.h5"),
                location=location,
            )
            for i, location in enumerate(("target", "host", "target"))
        ]

        with prepare_objects("keras") as (dataset, model_wrapper):
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

            assert model_path and model_path.exists()
            assert (
                model_path.read_bytes()
                == model_wrapper.model_path.read_bytes()
            )

    @pytest.mark.xdist_group(name="use_socket")
    @pytest.mark.parametrize("max_optimizers", (1, 2, 4, 8))
    def test_limit_target_side_optimization(self, max_optimizers: int):
        """
        Test various target-side compilation scenarios.
        """
        optimizers = [
            OptimizerMock.get_new_mock(
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
        protocol_target = NetworkProtocol(
            "localhost", 12345, timeout=TIMEOUT, packet_size=32768
        )
        inference_server = InferenceServer(
            runtime=runtime_target, protocol=protocol_target
        )

        with prepare_objects("keras") as (dataset, model_wrapper):
            dataconverter = ModelWrapperDataConverter(model_wrapper)
            runtime_host = TFLiteRuntime(
                model_path=Path("./build/compiled_model.tflite"),
            )
            protocol_host = NetworkProtocol(
                "localhost", 12345, timeout=TIMEOUT, packet_size=32768
            )

            pipeline_runner = PipelineRunner(
                dataset=dataset,
                dataconverter=dataconverter,
                model_wrapper=model_wrapper,
                optimizers=optimizers,
                runtime=runtime_host,
                protocol=protocol_host,
            )

            with running_inference_server(inference_server, protocol_host):
                pipeline_runner._handle_optimizations(
                    max_target_side_optimizers=max_optimizers
                )
                assert max_loaded_optimizers <= max_optimizers
