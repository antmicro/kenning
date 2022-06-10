"""
Runtime implementation for TVM-compiled models.
"""

from typing import Optional
from pathlib import Path
import numpy as np
from base64 import b64encode
import json

import tvm
from tvm.contrib import graph_executor
from tvm.runtime.vm import VirtualMachine, Executable

from kenning.core.runtime import Runtime
from kenning.core.runtimeprotocol import RuntimeProtocol
from kenning.core.runtimeprotocol import MessageType


class TVMRuntime(Runtime):

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
        'inputdtype': {
            'argparse_name': '--input-dtype',
            'description': 'Type of input tensor elements',
            'type': str,
            'default': 'float32'
        },
        'use_tvm_vm': {
            'argparse_name': '--runtime-use-vm',
            'description': 'At runtime use the TVM Relay VirtualMachine',
            'type': bool,
            'default': False
        },
        'use_json_out': {
            'argparse_name': '--use-json-at-output',
            'description': 'Encode outputs of models into a JSON file with base64-encoded arrays',  # noqa: E501
            'type': bool,
            'default': False
        },
        'io_details_path': {
            'description': "Path where the quantization details are saved in json. \
                By default <save_model_path>.quantparams is checked",
            'type': Path,
            'required': False
        }
    }

    def __init__(
            self,
            protocol: RuntimeProtocol,
            modelpath: Path,
            contextname: str = 'cpu',
            contextid: int = 0,
            inputdtype: str = 'float32',
            use_tvm_vm: bool = False,
            use_json_out: bool = False,
            io_details_path: Optional[Path] = None,
            collect_performance_data: bool = True):
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
        io_details_path : Optional[Path]
            Path for the quantization details file generated
            by tflite optimizer. Can be None.
        """
        self.modelpath = modelpath
        self.contextname = contextname
        self.contextid = contextid
        self.inputdtype = inputdtype
        self.model_inputdtype = inputdtype
        self.input_details = None
        self.output_details = None
        self.module = None
        self.func = None
        self.ctx = None
        self.model = None
        self.use_tvm_vm = use_tvm_vm
        self.use_json_out = use_json_out
        self.io_details_path = io_details_path
        super().__init__(protocol, collect_performance_data)
        self.callbacks[MessageType.QUANTIZATION] = \
            self._prepare_quantization_details

    @classmethod
    def from_argparse(cls, protocol, args):
        return cls(
            protocol,
            args.save_model_path,
            args.target_device_context,
            args.target_device_context_id,
            args.input_dtype,
            args.runtime_use_vm,
            args.use_json_at_output,
            args.io_details_path
        )

    def upload_essentials(self, compiledmodelpath):
        super().upload_essentials(compiledmodelpath)
        self.upload_quantization_details(compiledmodelpath)

    def prepare_local(self):
        super().prepare_local()
        self.prepare_quantization_details(None)

    def get_quantization_details_path(
            self,
            modelpath: Path) -> Path:
        """
        Gets path to a preferred quantization details file.

        If ``self.io_details_path`` is not specified, then the preferred path
        is ``self.modelpath`` with '.quantparams' suffix.

        Parameters
        ----------
        modelpath : Path
            Path to the compiled model

        Returns
        -------
        Path : Returns preferred path to a quantization details file
        """
        if self.io_details_path:
            path = self.io_details_path
        else:
            name = modelpath.stem.split('.')[0]
            parent = modelpath.parent
            path = parent.joinpath(name).with_suffix('.quantparams')

        return path

    def _prepare_quantization_details(
            self,
            input_data: Optional[bytes]) -> bool:
        """
        Wrapper for preparing quantization details.

        Parameters
        ----------
        input_data : Optional[bytes]
            Quantization details data or None, if the data should be loaded
            from another source.

        Returns
        -------
        bool : True if there is no data to send or if succeded
        """
        ret = self.prepare_quantization_details(input_data)
        if ret:
            self.protocol.request_success()
        else:
            self.protocol.request_failure()
        return ret

    def prepare_quantization_details(self, input_data):
        if input_data:
            self.input_details, self.output_details = json.loads(input_data)
        else:
            path = self.get_quantization_details_path(self.modelpath)

            if not path.exists():
                if self.io_details_path:
                    self.log.info(
                        f'Could not find specified quantization details \
                        file {path}'
                    )
                    return False

                # The path was not specified by the user
                # and there is no file in the preferred path
                return True

            with open(path, 'rb') as f:
                self.input_details, self.output_details = json.load(f)

        self.input_details = self.input_details[0]
        self.output_details = self.output_details[0]
        self.model_inputdtype = self.input_details['dtype']
        self.log.info('Quantization details loaded')
        return True

    def upload_quantization_details(self, compiledmodelpath):
        path = self.get_quantization_details_path(compiledmodelpath)

        if path.exists():
            self.protocol.upload_quantization_details(path)

    def prepare_input(self, input_data):
        self.log.debug(f'Preparing inputs of size {len(input_data)}')

        input_data = np.frombuffer(input_data, dtype=self.inputdtype)
        if self.input_details:
            if self.model_inputdtype != 'float32':
                scale, zero_point = self.input_details['quantization']
                input_data = input_data / scale + zero_point

                input_data = input_data.astype(self.model_inputdtype)

        try:
            if self.use_tvm_vm:
                self.model.set_input(
                    "main",
                    [tvm.nd.array(input_data)]
                )
            else:
                self.model.set_input(
                    0,
                    tvm.nd.array(input_data)
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

        dequantize_output = (
            self.output_details and
            self.model_inputdtype != 'float32'
        )

        if dequantize_output:
            scale, zero_point = self.output_details['quantization']

        def convert(output):
            if dequantize_output:
                return (
                        (output.astype(self.inputdtype) - zero_point)
                        * scale
                    ).tobytes()
            else:
                return output.tobytes()

        if self.use_tvm_vm:
            if self.use_json_out:
                out_dict = {}
                for i in range(len(self.model.get_outputs())):
                    out_dict[i] = b64encode(
                        convert(self.model.get_outputs()[i].asnumpy())
                    ).decode("ascii")
                json_str = json.dumps(out_dict)
                out = bytes(json_str, "ascii")
            else:
                for i in range(len(self.model.get_outputs())):
                    out += convert(self.model.get_outputs()[i].asnumpy())
        else:
            for i in range(self.model.get_num_outputs()):
                out += convert(self.model.get_output(i).asnumpy())

        return out
