# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime implementation for TFLite models.
"""

from typing import Optional, List

from kenning.core.runtime import Runtime
from kenning.core.runtime import ModelNotPreparedError
from kenning.core.runtime import InputNotPreparedError
from kenning.core.runtimeprotocol import RuntimeProtocol
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class TFLiteRuntime(Runtime):
    """
    Runtime subclass that provides an API
    for testing inference on TFLite models.
    """

    inputtypes = ['tflite']

    arguments_structure = {
        'model_path': {
            'argparse_name': '--save-model-path',
            'description': 'Path where the model will be uploaded',
            'type': ResourceURI,
            'default': 'model.tar'
        },
        'delegates': {
            'argparse_name': '--delegates-list',
            'description': 'List of runtime delegates for the TFLite runtime',
            'default': None,
            'is_list': True,
            'nullable': True
        },
        'num_threads': {
            'description': 'Number of threads to use for inference',
            'default': 4,
            'type': int
        }
    }

    def __init__(
            self,
            protocol: RuntimeProtocol,
            model_path: PathOrURI,
            delegates: Optional[List] = None,
            num_threads: int = 4,
            disable_performance_measurements: bool = False):
        """
        Constructs TFLite Runtime pipeline.

        Parameters
        ----------
        protocol : RuntimeProtocol
            The implementation of the host-target communication  protocol.
        model_path : PathOrURI
            Path or URI to the model file.
        delegates : Optional[List]
            List of TFLite acceleration delegate libraries.
        num_threads : int
            Number of threads to use for inference.
        disable_performance_measurements : bool
            Disable collection and processing of performance metrics.
        """
        self.model_path = model_path
        self.interpreter = None
        self._input_prepared = False
        self.num_threads = num_threads
        self.delegates = delegates
        super().__init__(
            protocol,
            disable_performance_measurements
        )

    def prepare_model(self, input_data):
        try:
            import tflite_runtime.interpreter as tflite
        except ModuleNotFoundError:
            from tensorflow import lite as tflite
        self.log.info('Loading model')
        if input_data:
            with open(self.model_path, 'wb') as outmodel:
                outmodel.write(input_data)
        else:
            self.model_path
        delegates = None
        if self.delegates:
            delegates = [tflite.load_delegate(delegate) for delegate in self.delegates]  # noqa: E501
        self.interpreter = tflite.Interpreter(
            str(self.model_path),
            experimental_delegates=delegates,
            num_threads=self.num_threads
        )
        self.interpreter.allocate_tensors()
        self.log.info('Model loading ended successfully')
        return True

    def prepare_input(self, input_data):
        self.log.debug(f'Preparing inputs of size {len(input_data)}')
        if self.interpreter is None:
            raise ModelNotPreparedError

        try:
            ordered_input = self.preprocess_input(input_data)
            for det, inp in zip(self.interpreter.get_input_details(), ordered_input):  # noqa: E501
                self.interpreter.set_tensor(det['index'], inp)
        except ValueError as ex:
            self.log.error(f'Failed to load input: {ex}')
            return False
        self._input_prepared = True
        return True

    def run(self):
        if self.interpreter is None:
            raise ModelNotPreparedError
        if not self._input_prepared:
            raise InputNotPreparedError
        self.interpreter.invoke()

    def extract_output(self):
        if self.interpreter is None:
            raise ModelNotPreparedError

        results = []
        for det in self.interpreter.get_output_details():
            out = self.interpreter.tensor(det['index'])()
            results.append(out.copy())

        return self.postprocess_output(results)
