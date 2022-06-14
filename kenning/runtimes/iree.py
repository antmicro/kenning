"""
Runtime implementation for IREE models
"""

from pathlib import Path
import numpy as np
from iree import runtime as ireert

from kenning.core.runtime import Runtime
from kenning.core.runtimeprotocol import RuntimeProtocol


# TODO: add support for multi-input/multi-output models
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
        # TODO: multi-input models
        dtype, shape = self.dtype[0], self.shapes[0]

        self.input = np.frombuffer(input_data, dtype=dtype)
        self.input = self.input.reshape(*shape)
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
        # TODO: get information regarding 'v'
        # input_signatures == [['ndarray', dtype, rank, *shape], ...]
        # dtype is represented as 'f32', 'i16' etc. Conversion to 'float32', 'int16' is required

        self.shapes = [sign[3:] for sign in input_signatures]
        encoded_dtypes = [sign[1] for sign in input_signatures]
        self.dtype = []
        dtype_codes = {'f': 'float'}  # TODO: add more dtypes
        for dtype in encoded_dtypes:
            self.dtype.append(dtype_codes[dtype[0]] + dtype[1:])

        self.log.info('Model loading ended successfully')
        return True

    def run(self):
        self.output = self.model.predict(self.input)

    def upload_output(self, input_data):
        return self.output.to_host().tobytes()
