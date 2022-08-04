"""
Runtime implementation for TVM-compiled models.
"""

from pathlib import Path
import numpy as np

import tvm
from tvm.contrib import graph_executor
from tvm.runtime.vm import VirtualMachine, Executable

from kenning.core.runtime import Runtime
from kenning.core.runtimeprotocol import RuntimeProtocol


class TVMRuntime(Runtime):
    """
    Runtime subclass that provides an API
    for testing inference on TVM models.
    """

    arguments_structure = {
        'modelpath': {
            'argparse_name': '--save-model-path',
            'description': 'Path where the model will be uploaded',
            'type': Path,
            'default': 'model.tar'
        },
        'contextname': {
            'argparse_name': '--target-device-context',
            'description': 'What accelerator should be used on target device',
            'default': 'cpu',
            'enum': list(tvm.runtime.Device.STR2MASK.keys())
        },
        'contextid': {
            'argparse_name': '--target-device-context-id',
            'description': 'ID of the device to run the inference on',
            'type': int,
            'default': 0
        },
        'use_tvm_vm': {
            'argparse_name': '--runtime-use-vm',
            'description': 'At runtime use the TVM Relay VirtualMachine',
            'type': bool,
            'default': False
        }
    }

    def __init__(
            self,
            protocol: RuntimeProtocol,
            modelpath: Path,
            contextname: str = 'cpu',
            contextid: int = 0,
            use_tvm_vm: bool = False,
            collect_performance_data: bool = True):
        """
        Constructs TVM runtime.

        Parameters
        ----------
        protocol : RuntimeProtocol
            The implementation of the host-target communication  protocol
        modelpath : Path
            Path for the model file.
        contextname : str
            Name of the runtime context on the target device
        contextid : int
            ID of the runtime context device
        use_tvm_vm : bool
            Use the TVM Relay VirtualMachine
        collect_performance_data : bool
            Disable collection and processing of performance metrics
        """
        self.modelpath = modelpath
        self.contextname = contextname
        self.contextid = contextid
        self.module = None
        self.func = None
        self.ctx = None
        self.model = None
        self.use_tvm_vm = use_tvm_vm
        super().__init__(
            protocol,
            collect_performance_data
        )

    @classmethod
    def from_argparse(cls, protocol, args):
        return cls(
            protocol,
            args.save_model_path,
            args.target_device_context,
            args.target_device_context_id,
            args.runtime_use_vm,
            args.disable_performance_measurements
        )

    def prepare_input(self, input_data):
        self.log.debug(f'Preparing inputs of size {len(input_data)}')
        input = {}

        # TODO: Check for a quantization

        for spec in self.input_spec:
            shape = spec['shape']
            dt = np.dtype(spec['dtype'])

            siz = np.abs(np.prod(shape) * dt.itemsize)
            inp = np.frombuffer(input_data[:siz], dtype=dt)
            inp = inp.reshape(shape)

            # if self.model_inputdtype != np.float32:
            #     scale = properties['scale']
            #     zero_point = properties['zero_point']
            #     inp = inp / scale + zero_point

            input[spec['name']] = tvm.nd.array(
                inp.astype(dt).reshape(shape)
            )
            input_data = input_data[siz:]

        try:
            if self.use_tvm_vm:
                self.model.set_input(
                    "main",
                    **input
                )
            else:
                self.model.set_input(
                    **input
                )
            self.log.debug('Inputs are ready')
            return True
        except (TypeError, ValueError, tvm.TVMError) as ex:
            self.log.error(f'Failed to load input:  {ex}')
            return False

    def prepare_model(self, input_data):
        self.log.info('Loading model')
        if self.use_tvm_vm:
            self.module = tvm.runtime.load_module(str(self.modelpath)+'.so')
            loaded_bytecode = bytearray(
                open(str(self.modelpath)+'.ro', "rb").read()
            )
            loaded_vm_exec = Executable.load_exec(loaded_bytecode, self.module)

            self.ctx = tvm.cpu()

            self.model = VirtualMachine(loaded_vm_exec, self.ctx)
        else:
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

        # TODO: Check for a quantization

        def convert(output):
            return output.tobytes()

        if self.use_tvm_vm:
            for output in self.model.get_outputs():
                out += convert(output.asnumpy())
        else:
            for i in range(self.model.get_num_outputs()):
                out += convert(self.model.get_output(i).asnumpy())

        return out
