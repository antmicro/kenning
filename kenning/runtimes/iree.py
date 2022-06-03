"""
Runtime implementation for IREE models
"""

from pathlib import Path
import numpy as np
from iree import runtime as ireert
import functools
import operator as op

from kenning.core.runtime import Runtime
from kenning.core.runtimeprotocol import RuntimeProtocol


# TODO: int dtype support

class IREERuntime(Runtime):

    arguments_structure = {
        'modelpath': {
            'argparse_name': '--save-model-path',
            'description': 'Path where the model will be uploaded',
            'type': Path,
            'default': 'model.vmfb'
        },
        'backend': {
            'argaprse_name': '--driver',
            'description': 'Name of the runtime target',
            'enum': ireert.HalDriver.query(),
            'required': True
        }
    }

    def __init__(
            self,
            protocol: RuntimeProtocol,
            modelpath: str,
            driver: str,
            collect_performance_data: bool = True):
        """
        Constructs IREE runtime

        Parameters
        ----------
        protocol : RuntimeProtocol
            Communication protocol
        modelpath : Path
            Path for the model file.
        driver : str
            Name of the deployment target on the device
        """
        self.modelpath = modelpath
        self.driver = driver
        super().__init__(protocol, collect_performance_data)

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
        for shape, dtype in zip(self.shapes, self.dtypes):
            dt = np.dtype(dtype)
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
        self.model = ireert.load_vm_flatbuffer_file(self.modelpath, driver=self.driver)
        module_function = self.model.vm_module.lookup_function("predict")
        input_signatures = eval(module_function.reflection['iree.abi'])['a']

        # reflection provides information regarding input ('a'), output ('r'), and a 'v' key,
        # input_signatures == [['ndarray', dtype, rank, *shape], ...]
        # output signature may be ['ndarray', ...] or [['stuple', ['ndarray', ...], ...] in case of multiple outputs
        # dtype is represented as 'f32', 'i16' etc. Conversion to 'float32', 'int16' is required

        self.shapes = [sign[3:] for sign in input_signatures]
        encoded_dtypes = [sign[1] for sign in input_signatures]
        self.dtypes = []
        dtype_codes = {'f': 'float'}  # TODO: add more dtypes
        for dtype in encoded_dtypes:
            self.dtypes.append(dtype_codes[dtype[0]] + dtype[1:])

        self.log.info('Model loading ended successfully')
        return True

    def run(self):
        self.output = self.model.predict(*self.input)

    def upload_output(self, input_data):
        try:
            return self.output.to_host().tobytes()
        except AttributeError:
            return functools.reduce(op.add, [out.to_host().tobytes() for out in self.output])
