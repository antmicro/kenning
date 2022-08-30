"""
Runtime implementation for TFLite models.
"""

from pathlib import Path
import numpy as np
from typing import Optional, List

from kenning.core.runtime import Runtime
from kenning.core.runtimeprotocol import RuntimeProtocol


class TFLiteRuntime(Runtime):
    """
    Runtime subclass that provides an API
    for testing inference on TFLite models.
    """

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
        }
    }

    def __init__(
            self,
            protocol: RuntimeProtocol,
            modelpath: Path,
            delegates: Optional[List] = None,
            collect_performance_data: bool = True):
        """
        Constructs TFLite Runtime pipeline.

        Parameters
        ----------
        protocol : RuntimeProtocol
            The implementation of the host-target communication  protocol
        modelpath : Path
            Path for the model file.
        delegates : List
            List of TFLite acceleration delegate libraries
        collect_performance_data : bool
            Disable collection and processing of performance metrics
        """
        self.modelpath = modelpath
        self.interpreter = None
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
            num_threads=4
        )
        self.interpreter.allocate_tensors()
        self.log.info('Model loading ended successfully')
        return True

    def prepare_input(self, input_data):
        self.log.debug(f'Preparing inputs of size {len(input_data)}')
        ordered_input = self.preprocess_input(input_data)

        for det, inp in zip(self.interpreter.get_input_details(), ordered_input):  # noqa: E501
            self.interpreter.tensor(det['index'])()[0] = inp
        return True

    def run(self):
        self.interpreter.invoke()

    def upload_output(self, input_data):
        self.log.debug('Uploading output')
        results = []
        dt = np.dtype(np.float32)
        for det in self.interpreter.get_output_details():
            out = self.interpreter.tensor(det['index'])()
            if det['dtype'] != np.float32:
                scale, zero_point = det['quantization']
                out = (out.astype(np.float32) - zero_point) * scale
                out = out.astype(dt)
            results.append(out.tobytes())

        return self.postprocess_output(results)
