"""
Runtime implementation for IREE models
"""

from pathlib import Path
import numpy as np
from iree import runtime as ireert
import functools
import ast
import operator as op

from kenning.core.runtime import Runtime
from kenning.core.runtimeprotocol import RuntimeProtocol


class IREERuntime(Runtime):
    """
    Runtime subclass that provides an API
    for testing inference on IREE models.
    """

    arguments_structure = {
        'modelpath': {
            'argparse_name': '--save-model-path',
            'description': 'Path where the model will be uploaded',
            'type': Path,
            'default': 'model.vmfb'
        },
        'backend': {
            'argparse_name': '--driver',
            'description': 'Name of the runtime target',
            'enum': ireert.HalDriver.query(),
            'required': True
        }
    }

    def __init__(
            self,
            protocol: RuntimeProtocol,
            modelpath: Path,
            driver: str,
            collect_performance_data: bool = True):
        """
        Constructs IREE runtime

        Parameters
        ----------
        protocol : RuntimeProtocol
            The implementation of the host-target communication protocol
        modelpath : Path
            Path for the model file.
        driver : str
            Name of the deployment target on the device
        collect_performance_data : bool
            Disable collection and processing of performance metrics
        """
        self.modelpath = modelpath
        self.driver = driver
        super().__init__(
            protocol,
            collect_performance_data
        )

    @classmethod
    def from_argparse(cls, protocol, args):
        return cls(
            protocol,
            args.save_model_path,
            args.driver,
            args.disable_performance_measurements
        )

    def prepare_input(self, input_data):
        self.input = []
        dt = np.dtype(self.dtype)
        for shape in self.shapes:
            siz = np.prod(shape) * dt.itemsize
            inp = np.frombuffer(input_data[:siz], dtype=dt)
            inp = inp.reshape(shape)
            self.input.append(inp)
            input_data = input_data[siz:]
        return True

    def prepare_model(self, input_data):
        self.log.info("loading model")
        if input_data:
            with open(self.modelpath, 'wb') as outmodel:
                outmodel.write(input_data)

        with open(self.modelpath, "rb") as outmodel:
            model_bytes = outmodel.read()
        model_dict = ast.literal_eval(model_bytes.decode("utf-8"))
        self.dtype = model_dict['dtype']
        self.shapes = model_dict['shapes']
        self.model = ireert.load_vm_flatbuffer(
            model_dict['model'], driver=self.driver)

        self.log.info('Model loading ended successfully')
        return True

    def run(self):
        self.output = self.model.main(*self.input)

    def upload_output(self, input_data):
        try:
            return self.output.to_host().tobytes()
        except AttributeError:
            return functools.reduce(
                op.add, [out.to_host().tobytes() for out in self.output])
