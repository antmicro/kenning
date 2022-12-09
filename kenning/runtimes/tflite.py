"""
Runtime implementation for TFLite models.
"""

from pathlib import Path
from typing import Optional, List

from kenning.core.runtime import Runtime
from kenning.core.runtime import ModelNotPreparedError
from kenning.core.runtime import InputNotPreparedError
from kenning.core.runtimeprotocol import RuntimeProtocol


class TFLiteRuntime(Runtime):
    """
    Runtime subclass that provides an API
    for testing inference on TFLite models.
    """

    inputtypes = ['tflite']

    arguments_structure = {
        'modelpath': {
            'argparse_name': '--save-model-path',
            'description': 'Path where the model will be uploaded',
            'type': Path,
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
            modelpath: Path,
            delegates: Optional[List] = None,
            num_threads: int = 4,
            collect_performance_data: bool = True):
        """
        Constructs TFLite Runtime pipeline.

        Parameters
        ----------
        protocol : RuntimeProtocol
            The implementation of the host-target communication  protocol
        modelpath : Path
            Path for the model file.
        delegates : Optional[List]
            List of TFLite acceleration delegate libraries
        num_threads : int
            Number of threads to use for inference
        collect_performance_data : bool
            Disable collection and processing of performance metrics
        """
        self.modelpath = modelpath
        self.interpreter = None
        self._input_prepared = False
        self.num_threads = num_threads
        self.delegates = delegates
        super().__init__(
            protocol,
            collect_performance_data
        )

    @classmethod
    def from_argparse(cls, protocol, args):
        return cls(
            protocol,
            args.save_model_path,
            args.delegates_list,
            args.num_threads,
            args.disable_performance_measurements
        )

    def prepare_model(self, input_data):
        try:
            import tflite_runtime.interpreter as tflite
        except ModuleNotFoundError:
            from tensorflow import lite as tflite
        self.log.info('Loading model')
        if input_data:
            with open(self.modelpath, 'wb') as outmodel:
                outmodel.write(input_data)
        delegates = None
        if self.delegates:
            delegates = [tflite.load_delegate(delegate) for delegate in self.delegates]  # noqa: E501
        self.interpreter = tflite.Interpreter(
            str(self.modelpath),
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

    def upload_output(self, input_data):
        self.log.debug('Uploading output')
        if self.interpreter is None:
            raise ModelNotPreparedError

        results = []
        for det in self.interpreter.get_output_details():
            out = self.interpreter.tensor(det['index'])()
            results.append(out)

        return self.postprocess_output(results)
