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

    supported_types = ['float32', 'int8', 'uint8']

    arguments_structure = {
        'modelpath': {
            'argparse_name': '--save-model-path',
            'description': 'Path where the model will be uploaded',
            'type': Path,
            'default': 'model.tar'
        },
        'inputdtype': {
            'argparse_name': '--input-dtype',
            'description': 'Type of input tensor elements',
            'enum': supported_types,
            'default': 'float32'
        },
        'outputdtype': {
            'argparse_name': '--output-dtype',
            'description': 'Type of output tensor elements',
            'enum': supported_types,
            'default': 'float32'
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
            inputdtype: str = 'float32',
            outputdtype: str = 'float32',
            delegates: Optional[List] = None,
            collect_performance_data: bool = True):
        """
        Constructs TFLite Runtime pipeline.

        Parameters
        ----------
        protocol : RuntimeProtocol
            Communication protocol
        modelpath : Path
            Path for the model file.
        inputdtype : str
            Type of the input data
        outputdtype : str
            Type of the output data
        delegates : List
            List of TFLite acceleration delegate libraries
        """
        self.modelpath = modelpath
        self.interpreter = None
        self.inputdtype = inputdtype
        self.outputdtype = outputdtype
        self.delegates = delegates
        super().__init__(protocol, collect_performance_data)

    @classmethod
    def from_argparse(cls, protocol, args):
        return cls(
            protocol,
            args.save_model_path,
            args.input_dtype,
            args.output_dtype,
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
        for model_details in self.interpreter.get_input_details():
            datatype = np.dtype(self.inputdtype)
            expected_size = np.prod(model_details['shape']) * datatype.itemsize
            input = np.frombuffer(input_data[:expected_size], dtype=datatype)
            try:
                input = input.reshape(model_details['shape'])
                input_size = np.prod(input.shape) * datatype.itemsize
                if expected_size != input_size:
                    self.log.error(f'Invalid input size:  {expected_size} != {input_size}')  # noqa E501
                    raise ValueError
                if model_details['dtype'] != np.float32:
                    scale, zero_point = model_details['quantization']
                    input = (input / scale + zero_point).astype(model_details['dtype'])  # noqa E501
                self.interpreter.tensor(model_details['index'])()[0] = input
                input_data = input_data[expected_size:]
            except ValueError as ex:
                self.log.error(f'Failed to load input: {ex}')
                return False
        return True

    def run(self):
        self.interpreter.invoke()

    def upload_output(self, input_data):
        self.log.debug('Uploading output')
        result = bytes()
        datatype = np.dtype(self.outputdtype)
        for model_details in self.interpreter.get_output_details():
            output = self.interpreter.tensor(model_details['index'])()
            if model_details['dtype'] != np.float32:
                scale, zero_point = model_details['quantization']
                output = (output.astype(np.float32) - zero_point) * scale
                output = output.astype(datatype)
            result += output.tobytes()
        return result
