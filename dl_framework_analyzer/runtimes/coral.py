"""
Runtime implementation for Google Coral boards (based on pycoral).
"""

from pathlib import Path
import numpy as np

from dl_framework_analyzer.core.runtime import Runtime
from dl_framework_analyzer.core.runtimeprotocol import RuntimeProtocol


class GoogleCoralRuntime(Runtime):
    def __init__(
            self,
            protocol: RuntimeProtocol,
            modelpath: Path,
            inputdtype: str = 'float32',
            outputdtype: str = 'float32'):
        """
        Constructs Google Coral EdgeTPU pipeline.

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
        """
        self.modelpath = modelpath
        self.interpreter = None
        self.inputdtype = inputdtype
        self.outputdtype = outputdtype
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
        return parser, group

    @classmethod
    def from_argparse(cls, protocol, args):
        return cls(
            protocol,
            args.save_model_path,
            args.input_dtype,
            args.output_dtype
        )

    def prepare_model(self, input_data):
        from pycoral.utils import edgetpu
        self.protocol.log.info('Loading model')
        with open(self.modelpath, 'wb') as outmodel:
            outmodel.write(input_data)
        self.interpreter = edgetpu.make_interpreter(str(self.modelpath))
        self.interpreter.allocate_tensors()
        self.protocol.log.info('Model loading ended successfully')
        return True

    def prepare_input(self, input_data):
        self.protocol.log.debug(f'Preparing inputs of size {len(input_data)}')
        for det in self.interpreter.get_input_details():
            dt = np.dtype(self.inputdtype)
            siz = np.prod(det['shape']) * dt.itemsize
            inp = np.frombuffer(input_data[:siz], dtype=dt)
            inp = inp.reshape(det['shape'])
            if det['dtype'] != np.float32:
                scale, zero_point = det['quantization']
                inp = inp / scale + zero_point
            self.interpreter.tensor(det['index'])()[0] = inp.astype(det['dtype'])
            input_data = input_data[siz:]
        return True

    def run(self):
        self.interpreter.invoke()

    def upload_output(self, input_data):
        self.protocol.log.debug('Uploading output')
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
