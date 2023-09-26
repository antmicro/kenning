# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import shutil
import threading
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional
from kenning.core.optimizer import Optimizer
from kenning.scenarios.inference_server import _InferenceServer
from kenning.utils.pipeline_runner import PipelineRunner
from kenning.utils.resource_manager import PathOrURI

import pytest

from kenning.core.protocol import RequestFailure
from kenning.protocols.network import NetworkProtocol
from kenning.runtimes.tflite import TFLiteRuntime
from kenning.tests.core.conftest import get_default_dataset_model
from kenning.utils.class_loader import load_class


class OptimizerMock(Optimizer):
    """
    Optimizer mock that only copies model.
    """

    inputtypes = {'keras': lambda: None}

    outputtypes = ['keras']

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None
    ):
        shutil.copy(input_model_path, self.compiled_model_path)

    def get_framework_and_version(self):
        return 'none', 0

    def to_json(self) -> Dict[str, Any]:
        ret = super().to_json()
        ret['type'] = (
            'kenning.tests.optimizers.test_target_side_optimization.'
            'OptimizerMock'
        )
        return ret

class TestServerSideOptimization:
    @pytest.mark.xdist_group(name='use_socket')
    @pytest.mark.xdist_group(name='use_resources')
    def test_local_optimization(self):
        """
        Test local compilation.
        """
        optimizers = [
            OptimizerMock(
                dataset=None,
                compiled_model_path=Path(f'./build/compiled_model_{i}.h5')
            )
            for i in range(3)
        ]

        runtime_host = TFLiteRuntime(
            model_path=Path('./build/compiled_model.tflite'),
        )

        dataset, model_wrapper = get_default_dataset_model('keras')

        pipeline_runner = PipelineRunner(
            dataset=dataset,
            model_wrapper=model_wrapper,
            optimizers=optimizers,
            runtime=runtime_host,
        )

        model_path = pipeline_runner.handle_optimizations()

        assert model_path.exists()

    @pytest.mark.xdist_group(name='use_socket')
    @pytest.mark.xdist_group(name='use_resources')
    @pytest.mark.parametrize(
        'optimizers_locations',
        (
            ('host',),
            ('target',),
            ('host', 'target'),
            ('target', 'host'),
            ('host', 'host'),
            ('target', 'target'),
            ('host', 'target', 'host', 'target'),
            (
                'host',
                'target',
                'target',
                'target',
                'target',
                'host',
                'host',
                'target',
                'target',
                'host',
            ),
        ),
    )
    def test_target_side_optimization(self, optimizers_locations: List[str]):
        """
        Test various target-side compilation scenarios.
        """
        optimizers = [
            OptimizerMock(
                dataset=None,
                compiled_model_path=Path(f'./build/compiled_model_{i}.h5'),
                location=location
            )
            for i, location in enumerate(optimizers_locations)
        ]

        runtime_target = TFLiteRuntime(
            model_path=Path('./build/compiled_model.tflite'),
        )
        protocol_target = NetworkProtocol('localhost', 12345, 32768)
        inference_server = _InferenceServer(
            runtime=runtime_target,
            protocol=protocol_target
        )

        dataset, model_wrapper = get_default_dataset_model('keras')
        runtime_host = TFLiteRuntime(
            model_path=Path('./build/compiled_model.tflite'),
        )
        protocol_host = NetworkProtocol('localhost', 12345, 32768)

        pipeline_runner = PipelineRunner(
            dataset=dataset,
            model_wrapper=model_wrapper,
            optimizers=optimizers,
            runtime=runtime_host,
            protocol=protocol_host,
        )

        server_thread = threading.Thread(target=inference_server.run)
        try:
            server_thread.start()
            sleep(.1)

            assert server_thread.is_alive()

            protocol_host.initialize_client()

            model_path = pipeline_runner.handle_optimizations()
            assert model_path.exists()

        finally:
            inference_server.close()
            server_thread.join()
            assert not server_thread.is_alive()


    def test_optimization_fail_when_server_is_not_running(self):
        """
        Test target side optimizations handling when the server is not running.
        """
        optimizers = [
            OptimizerMock(
                dataset=None,
                compiled_model_path=Path(f'./build/compiled_model_{i}.h5'),
                location=location
            )
            for i, location in enumerate(('target', 'host', 'target'))
        ]

        dataset, model_wrapper = get_default_dataset_model('keras')
        runtime = TFLiteRuntime(
            model_path=Path('./build/compiled_model.tflite'),
        )
        protocol = NetworkProtocol('localhost', 12345, 32768)

        pipeline_runner = PipelineRunner(
            dataset=dataset,
            model_wrapper=model_wrapper,
            optimizers=optimizers,
            runtime=runtime,
            protocol=protocol,
        )

        with pytest.raises(RequestFailure):
            pipeline_runner.handle_optimizations()
