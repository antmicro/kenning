import json
from pathlib import Path
import numpy as np

import tvm
from tvm.contrib import graph_runtime

from dl_framework_analyzer.core.runtime import Runtime
from dl_framework_analyzer.core.runtimeprotocol import RuntimeProtocol
from dl_framework_analyzer.core.measurements import MeasurementsCollector


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
        """
        self.modelpath = modelpath
        self.contextname = contextname
        self.contextid = contextid
        self.inputdtype = inputdtype
        self.model = None
        self.lastoutput = None
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
            choices=list(tvm.runtime.TVMContext.STR2MASK.keys()),
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
        self.protocol.log.debug(f'Preparing inputs of size {len(input_data)}')
        try:
            self.model.set_input(
                0,
                tvm.nd.array(np.frombuffer(input_data, dtype=self.inputdtype))
            )
            self.protocol.request_success()
            self.protocol.log.debug('Inputs are ready')
        except (TypeError, ValueError, tvm.TVMError):
            self.protocol.log.error('Failed to load input')
            self.protocol.request_failure()

    def prepare_model(self, input_data):
        self.protocol.log.info('Loading model')
        with open(self.modelpath, 'wb') as outmodel:
            outmodel.write(input_data)
        module = tvm.runtime.load_module(str(self.modelpath))
        func = module.get_function('default')
        ctx = tvm.runtime.context(self.contextname, self.contextid)
        self.model = graph_runtime.GraphModule(func(ctx))
        self.protocol.request_success()
        self.protocol.log.info('Model loading ended successfully')

    def process_input(self, input_data):
        self.protocol.log.debug('Processing input')
        self.protocol.request_success()
        self.model.run()
        self.protocol.request_success()
        self.protocol.log.debug('Input processed')
        self.lastoutput = self.model.get_output(0).asnumpy().tobytes()

    def upload_output(self, input_data):
        self.protocol.log.info('Uploading output')
        if self.lastoutput:
            self.protocol.request_success(self.lastoutput)
            self.lastoutput = None
        else:
            self.protocol.request_failure()

    def upload_stats(self, input_data):
        self.protocol.log.info('Uploading stats')
        stats = json.dumps(MeasurementsCollector.measurements)
        self.protocol.request_success(stats.encode('utf-8'))
