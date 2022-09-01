"""
Runtime implementation for IREE models
"""

from pathlib import Path
from iree import runtime as ireert

from kenning.core.runtime import Runtime, ModelNotLoadedError
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
        self.model = None
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
        self.log.debug(f'Preparing inputs of size {len(input_data)}')
        if self.model is None:
            raise ModelNotLoadedError("You must prepare the model before running it.")  # noqa: E501

        try:
            self.input = self.preprocess_input(input_data)
        except ValueError as ex:
            self.log.error(f'Failed to load input: {ex}')
            return False
        return True

    def prepare_model(self, input_data):
        self.log.info("loading model")
        if input_data:
            with open(self.modelpath, 'wb') as outmodel:
                outmodel.write(input_data)

        with open(self.modelpath, "rb") as outmodel:
            compiled_buffer = outmodel.read()

        self.model = ireert.load_vm_flatbuffer(
            compiled_buffer, driver=self.driver
        )

        self.log.info('Model loading ended successfully')
        return True

    def run(self):
        if self.model is None:
            raise ModelNotLoadedError("You must prepare the model before running it.")  # noqa: E501
        self.output = self.model.main(*self.input)

    def upload_output(self, input_data):
        self.log.debug('Uploading output')

        results = []
        try:
            results.append(self.output.to_host())
        except AttributeError:
            for out in self.output:
                results.append(out.to_host())

        return self.postprocess_output(results)
