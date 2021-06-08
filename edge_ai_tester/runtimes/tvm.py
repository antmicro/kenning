"""
Runtime implementation for TVM-compiled models.
"""

from pathlib import Path
import numpy as np

import tvm
from tvm.contrib import graph_executor

from edge_ai_tester.core.runtime import Runtime
from edge_ai_tester.core.runtimeprotocol import RuntimeProtocol


class TVMRuntime(Runtime):
    def __init__(
            self,
            protocol: RuntimeProtocol,
            modelpath: Path,
            contextname: str = 'cpu',
            contextid: int = 0,
            inputdtype: str = 'float32'):
        """
        Constructs TVM runtime.

        Parameters
        ----------
        protocol : RuntimeProtocol
            Communication protocol.
        modelpath : Path
            Path for the model file.
        contextname : str
            Name of the runtime context on the target device
        contextid : int
            ID of the runtime context device
        inputdtype : str
            Type of the input data
        """
        self.modelpath = modelpath
        self.contextname = contextname
        self.contextid = contextid
        self.inputdtype = inputdtype
        self.module = None
        self.func = None
        self.ctx = None
        self.model = None
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
            '--target-device-context',
            help='What accelerator should be used on target device',
            choices=list(tvm.runtime.Device.STR2MASK.keys()),
            default='cpu'
        )
        group.add_argument(
            '--target-device-context-id',
            help='ID of the device to run the inference on',
            type=int,
            default=0
        )
        group.add_argument(
            '--input-dtype',
            help='Type of input tensor elements',
            type=str,
            default='float32'
        )
        return parser, group

    @classmethod
    def from_argparse(cls, protocol, args):
        return cls(
            protocol,
            args.save_model_path,
            args.target_device_context,
            args.target_device_context_id,
            args.input_dtype
        )

    def prepare_input(self, input_data):
        self.log.debug(f'Preparing inputs of size {len(input_data)}')
        try:
            self.model.set_input(
                0,
                tvm.nd.array(np.frombuffer(input_data, dtype=self.inputdtype))
            )
            self.log.debug('Inputs are ready')
            return True
        except (TypeError, ValueError, tvm.TVMError) as ex:
            self.log.error(f'Failed to load input:  {ex}')
            return False

    def prepare_model(self, input_data):
        self.log.info('Loading model')
        if input_data:
            with open(self.modelpath, 'wb') as outmodel:
                outmodel.write(input_data)
        self.module = tvm.runtime.load_module(str(self.modelpath))
        self.func = self.module.get_function('default')
        self.ctx = tvm.runtime.device(self.contextname, self.contextid)
        self.model = graph_executor.GraphModule(self.func(self.ctx))
        self.log.info('Model loading ended successfully')
        return True

    def run(self):
        self.model.run()

    def upload_output(self, input_data):
        self.log.debug('Uploading output')
        out = b''
        for i in range(self.model.get_num_outputs()):
            out += self.model.get_output(i).asnumpy().tobytes()
        return out
