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
            The implementation of the host-target communication  protocol
        modelpath : Path
            Path for the model file.
        execution_providers : List[str]
            List of execution providers ordered by priority
        modelpath : Path
            Path for the model file.
        collect_performance_data : bool
            Disable collection and processing of performance metrics
        """
        self.modelpath = modelpath
        self.execution_providers = execution_providers
        super().__init__(
            protocol,
            collect_performance_data
        )

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
        ordered_input = self.preprocess_input(input_data)
        self.input = {}

        for spec, inp in zip(self.input_spec, ordered_input):
            self.input[spec['name']] = inp
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

        def update_io_spec(read_spec, session_spec):
            model_spec = []
            for input in session_spec:
                model_spec.append({
                    'name': input.name,
                    'shape': np.array([s if isinstance(s, int) else -1 for s in input.shape]),  # noqa: E501
                    'dtype': onnx_to_np_dtype(input.type)
                })

            if not read_spec:
                return model_spec
            else:
                for s, m in zip(read_spec, model_spec):
                    if 'name' not in s:
                        s['name'] = m['name']
                    if 'shape' not in s:
                        s['shape'] = m['shape']
                    if 'dtype' not in s:
                        s['dtype'] = m['dtype']

            return read_spec

        self.input_spec = update_io_spec(
            self.input_spec,
            self.session.get_inputs()
        )

        self.output_spec = update_io_spec(
            self.output_spec,
            self.session.get_outputs()
        )

        self.log.info('Model loading ended successfully')
        return True

    def run(self):
        self.scores = self.session.run(
            [spec['name'] for spec in self.output_spec],
            self.input
        )

    def upload_output(self, input_data):
        self.log.debug('Uploading output')
        results = []

        for i in range(len(self.session.get_outputs())):
            results.append(self.scores[i])

        return self.postprocess_output(results)
