"""
Runtime implementation for ONNX models.
"""

from typing import List
import onnxruntime as ort
from pathlib import Path
import numpy as np

from kenning.core.runtime import Runtime
from kenning.core.runtimeprotocol import RuntimeProtocol


class ONNXRuntime(Runtime):
    """
    Runtime subclass that provides an API
    for testing inference on ONNX models.
    """

    arguments_structure = {
        'modelpath': {
            'argparse_name': '--save-model-path',
            'description': 'Path where the model will be uploaded',
            'type': Path,
            'default': 'model.tar'
        },
        'execution_providers': {
            'description': 'List of execution providers ordered by priority',
            'is_list': True,
            'default': ['CPUExecutionProvider']
        }
    }

    def __init__(
            self,
            protocol: RuntimeProtocol,
            modelpath: Path,
            execution_providers: List[str] = ['CPUExecutionProvider'],
            collect_performance_data: bool = True):
        """
        Constructs ONNX runtime

        Parameters
        ----------
        protocol : RuntimeProtocol
            Communication protocol
        execution_providers : List[str]
            List of execution providers ordered by priority
        modelpath : Path
            Path for the model file.
        """

        self.modelpath = modelpath
        self.execution_providers = execution_providers
        super().__init__(protocol, collect_performance_data)

    @classmethod
    def from_argparse(cls, protocol, args):
        return cls(
            protocol,
            args.save_model_path,
            args.execution_providers,
            args.disable_performance_measurements
        )

    def prepare_input(self, input_data):
        self.log.debug(f'Preparing inputs of size {len(input_data)}')
        self.input = {}

        for dt, shape, name in zip(
                self.input_dtypes,
                self.input_shapes,
                self.input_names):

            siz = np.abs(np.prod(shape) * dt.itemsize)
            inp = np.frombuffer(input_data[:siz], dtype=dt)
            inp = inp.reshape(shape)
            self.input[name] = inp
            input_data = input_data[siz:]
        return True

    def prepare_model(self, input_data):
        self.log.info('Loading model')
        if input_data:
            with open(self.modelpath, 'wb') as outmodel:
                outmodel.write(input_data)

        self.session = ort.InferenceSession(
            str(self.modelpath),
            providers=self.execution_providers
        )

        # Input dtype can come either as a valid np.dtype
        # or as a string that need to be parsed
        def onnx_to_np_dtype(s):
            if not isinstance(s, str):
                return s
            if s == 'tensor(float)':
                return np.dtype(np.float32)
            if s == 'tensor(float16)':
                return np.dtype(np.float16)

        self.input_dtypes = [
            onnx_to_np_dtype(input.type) for input in self.session.get_inputs()
        ]
        self.input_shapes = [
            np.array([s if isinstance(s, int) else -1 for s in input.shape])
            for input in self.session.get_inputs()
        ]
        self.input_names = [
            input.name for input in self.session.get_inputs()
        ]
        self.output_names = [
            output.name for output in self.session.get_outputs()
        ]

        self.log.info('Model loading ended successfully')
        return True

    def run(self):
        self.scores = self.session.run(
            self.output_names,
            self.input
        )

    def upload_output(self, input_data):
        self.log.debug('Uploading output')
        result = bytes()

        for i, _ in enumerate(self.session.get_outputs()):
            result += self.scores[i].tobytes()

        return result
