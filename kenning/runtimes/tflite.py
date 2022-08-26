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
            'type': str,
            'default': ['float32'],
            'is_list': True
        },
        'outputdtype': {
            'argparse_name': '--output-dtype',
            'description': 'Type of output tensor elements',
            'enum': supported_types,
            'default': ['float32'],
            'is_list': True
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
            inputdtype: List[str] = ('float32',),
            outputdtype: List[str] = ('float32',),
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
        inputdtype : List[str]
            Type of the input data
        outputdtype : List[str]
            Type of the output data
        delegates : List
            List of TFLite acceleration delegate libraries
        """
        self.modelpath = modelpath
        self.signature = None
        self.inputdtype = inputdtype
        self.outputdtype = outputdtype
        self.delegates = delegates
        self.inputs = None
        self.outputs = None
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
        interpreter = tflite.Interpreter(
            str(self.modelpath),
            experimental_delegates=delegates,
            num_threads=4
        )
        interpreter.allocate_tensors()
        self.signature = interpreter.get_signature_runner()
        self.signatureinfo = interpreter.get_signature_list()['serving_default']  # noqa: E501

        self.outputdtype = [np.dtype(dt) for dt in self.outputdtype]
        self.inputdtype = [np.dtype(dt) for dt in self.inputdtype]

        self.log.info('Model loading ended successfully')
        return True

    def prepare_input(self, input_data):
        self.log.debug(f'Preparing inputs of size {len(input_data)}')
        self.inputs = {}
        input_names = self.signatureinfo['inputs']
        for datatype, name in zip(self.inputdtype, input_names):
            model_details = self.signature.get_input_details()[name]
            expected_size = np.prod(model_details['shape']) * datatype.itemsize
            input = np.frombuffer(input_data[:expected_size], dtype=datatype)
            try:
                input = input.reshape(model_details['shape'])
                input_size = np.prod(input.shape) * datatype.itemsize
                if expected_size != input_size:
                    self.log.error(f'Invalid input size:  {expected_size} != {input_size}')  # noqa E501
                    raise ValueError
                scale, zero_point = model_details['quantization']
                if scale != 0:
                    input = (input / scale + zero_point)
                self.inputs[name] = input.astype(model_details['dtype'])
                input_data = input_data[expected_size:]
            except ValueError as ex:
                self.log.error(f'Failed to load input: {ex}')
                return False
        if input_data:
            self.log.error("Failed to load input: Received more data than model expected.")  # noqa: E501
            return False
        return True

    def run(self):
        if self.signature is None:
            raise AttributeError("You must prepare the model before running it.")  # noqa: E501
        if self.inputs is not None:
            self.outputs = self.signature(**self.inputs)
            self.inputs = None

    def upload_output(self, input_data):
        self.log.debug('Uploading output')
        if self.outputs is None:
            raise AttributeError("No outputs were found ")
        result = bytes()
        output_names = self.signatureinfo['outputs']
        for datatype, name in zip(self.outputdtype, output_names):
            model_details = self.signature.get_output_details()[name]
            output = self.outputs[name]
            if datatype != model_details['dtype']:
                scale, zero_point = model_details['quantization']
                if scale != 0:
                    output = (output.astype(np.float32) - zero_point) * scale
                output = output.astype(datatype)
            result += output.tobytes()
        self.outputs = None
        return result
