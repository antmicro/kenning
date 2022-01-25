"""
Runtime implementation for TFLite models.
"""

from pathlib import Path
import numpy as np
from typing import Optional, List

from kenning.core.runtime import Runtime
from kenning.core.runtimeprotocol import RuntimeProtocol


class TFLiteRuntime(Runtime):
    def __init__(
            self,
            protocol: RuntimeProtocol,
            modelpath: Path,
            inputdtype: str = 'float32',
            outputdtype: str = 'float32',
            delegates: Optional[List] = None):
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
        super().__init__(protocol)

    @classmethod
    def form_argparse(cls):
        parser, group = super().form_argparse()
        group.add_argument(
            '--save-model-path',
            help='Path where the model will be uploaded',
            type=Path,
            default='model.tar'
        )
        group.add_argument(
            '--input-dtype',
            help='Type of input tensor elements',
            type=str,
            default='float32'
        )
        group.add_argument(
            '--output-dtype',
            help='Type of output tensor elements',
            type=str,
            default='float32'
        )
        group.add_argument(
            '--delegates-list',
            help='List of runtime delegates for the TFLite runtime',
            nargs='+',
            default=None
        )
        return parser, group

    @classmethod
    def from_argparse(cls, protocol, args):
        return cls(
            protocol,
            args.save_model_path,
            args.input_dtype,
            args.output_dtype,
            args.delegates_list
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
        for det in self.interpreter.get_input_details():
            dt = np.dtype(self.inputdtype)
            siz = np.prod(det['shape']) * dt.itemsize
            inp = np.frombuffer(input_data[:siz], dtype=dt)
            inp = inp.reshape(det['shape'])
            if det['dtype'] != np.float32:
                scale, zero_point = det['quantization']
                inp = inp / scale + zero_point
            self.interpreter.tensor(det['index'])()[0] = inp.astype(det['dtype'])  # noqa: E501
            input_data = input_data[siz:]
        return True

    def run(self):
        self.interpreter.invoke()

    def upload_output(self, input_data):
        self.log.debug('Uploading output')
        result = bytes()
        dt = np.dtype(self.outputdtype)
        for det in self.interpreter.get_output_details():
            out = self.interpreter.tensor(det['index'])()
            if det['dtype'] != np.float32:
                scale, zero_point = det['quantization']
                out = (out.astype(np.float32) - zero_point) * scale
                out = out.astype(dt)
            result += out.tobytes()
        return result
