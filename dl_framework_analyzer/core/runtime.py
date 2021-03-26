"""
Module providing a Runtime wrapper.

Runtimes implement running and testing deployed models on target devices.
"""

import argparse
from tqdm import tqdm
from pathlib import Path

from dl_framework_analyzer.core.dataset import Dataset
from dl_framework_analyzer.core.model import ModelWrapper
from dl_framework_analyzer.core.runtimeprotocol import RuntimeProtocol
from dl_framework_analyzer.core.runtimeprotocol import MessageType
from dl_framework_analyzer.core.runtimeprotocol import RequestFailure
from dl_framework_analyzer.core.runtimeprotocol import check_request
from dl_framework_analyzer.core.measurements import Measurements
from dl_framework_analyzer.core.measurements import MeasurementsCollector
from dl_framework_analyzer.core.runtimeprotocol import ServerStatus
from dl_framework_analyzer.core.measurements import timemeasurements
from dl_framework_analyzer.core.measurements import SystemStatsCollector


class Runtime(object):
    """
    Runtime object provides an API for testing inference on target devices.

    Using a provided RuntimeProtocol it sets up a client (host) and server
    (target) communication, during which the inference metrics are being
    analyzed.
    """

    def __init__(
            self,
            protocol: RuntimeProtocol):
        """
        Creates Runtime object.

        Parameters
        ----------
        protocol : RuntimeProtocol
            The implementation of the host-target communication  protocol
        """
        self.protocol = protocol
        self.shouldwork = True
        self.callbacks = {
            MessageType.DATA: self.prepare_input,
            MessageType.MODEL: self._prepare_model,
            MessageType.PROCESS: self.process_input,
            MessageType.OUTPUT: self.upload_output,
            MessageType.STATS: self._upload_stats
        }
        self.statsmeasurements = None

    @classmethod
    def form_argparse(cls):
        """
        Creates argparse parser for the Runtime object.

        Returns
        -------
        (ArgumentParser, ArgumentGroup) :
            tuple with the argument parser object that can act as parent for
            program's argument parser, and the corresponding arguments' group
            pointer
        """
        parser = argparse.ArgumentParser(add_help=False)
        group = parser.add_argument_group(title='Runtime arguments')
        return parser, group

    @classmethod
    def from_argparse(cls, protocol, args):
        """
        Constructor wrapper that takes the parameters from argparse args.

        Parameters
        ----------
        protocol : RuntimeProtocol object
        args : arguments from RuntimeProtocol object

        Returns
        -------
        RuntimeProtocol : object of class ModelCompiler
        """
        return cls(protocol)

    def inference_session_start(self):
        """
        Calling this function indicates that the client is connected.

        This method should be called once the client has connected to a server.

        This will enable performance tracking.
        """
        if self.statsmeasurements is None:
            self.statsmeasurements = SystemStatsCollector(
                'session_utilization'
            )
        self.statsmeasurements.start()

    def inference_session_end(self):
        """
        Calling this function indicates that the inference session has ended.

        This method should be called once all the inference data is sent to
        the server by the client.

        This will stop performance tracking.
        """
        if self.statsmeasurements:
            self.statsmeasurements.stop()
            self.statsmeasurements.join()
            MeasurementsCollector.measurements += \
                self.statsmeasurements.get_measurements()
            self.statsmeasurements = None

    def close_server(self):
        """
        Indicates that the server should be closed.
        """
        self.shouldwork = False

    def prepare_server(self):
        """
        Runs initialization of the server.
        """
        self.protocol.initialize_server()

    def prepare_client(self):
        """
        Runs initialization for the client.
        """
        self.protocol.initialize_client()

    def prepare_input(self, input_data: bytes):
        """
        Loads and converts delivered data to the accelerator for inference.

        This method is called when the input is received from the client.
        It is supposed to prepare input before running inference.

        Parameters
        ----------
        input_data : bytes
            Input data in bytes delivered by the client, preprocessed
        """
        raise NotImplementedError

    def _prepare_model(self, input_data: bytes):
        """
        Internal call for preparing a model for inference task.
        """
        self.inference_session_start()
        self.prepare_model(input_data)

    def prepare_model(self, input_data: bytes):
        """
        Receives the model to infer from the client in bytes.

        The method should load bytes with the model, optionally save to file
        and allocate the model on target device for inference.

        Parameters
        ----------
        input_data : bytes
            Model data
        """
        raise NotImplementedError

    def process_input(self, input_data):
        """
        Processes received input and measures the performance quality.

        Parameters
        ----------
        input_data : bytes
            Not used here
        """
        self.protocol.log.debug('Processing input')
        self.protocol.request_success()
        self._run()
        self.protocol.request_success()
        self.protocol.log.debug('Input processed')
        self.lastoutput = self.model.get_output(0).asnumpy().tobytes()

    @timemeasurements('target_inference_step')
    def _run(self):
        """
        Performance wrapper for run method.
        """
        self.run()

    def run(self):
        """
        Runs inference on prepared input.

        The input should be introduced in runtime's model representation, or
        it should be delivered using a variable that was assigned in
        prepare_input method.
        """
        raise NotImplementedError

    def upload_output(self, input_data: bytes):
        """
        Uploads the output to the client, in bytes.

        The method converts the direct output from the model to bytes and sends
        back to the client.

        Parameters
        ----------
        input_data : bytes
            Not used here
        """
        raise NotImplementedError

    def _upload_stats(self, input_data: bytes):
        """
        Wrapper for uploading stats.

        Stops measurements and uploads stats.

        Parameters
        ----------
        input_data : bytes
            Not used here
        """
        self.inference_session_end()
        self.upload_stats(input_data)

    def upload_stats(self, input_data: bytes):
        """
        Uploads statistics of inference passes to the client.

        Parameters
        ----------
        input_data : bytes
            Not used here
        """
        raise NotImplementedError

    def run_client(
            self,
            dataset: Dataset,
            modelwrapper: ModelWrapper,
            compiledmodelpath: Path):
        """
        Main runtime client program.

        The client performance procedure is as follows:

        * connect with the server
        * upload the model
        * send dataset data in a loop to the server:

            * upload input
            * request processing of inputs
            * request predictions for inputs
            * evaluate the response
        * collect performance statistics
        * end connection

        Parameters
        ----------
        dataset : Dataset
            Dataset to verify the inference on
        modelwrapper : ModelWrapper
            Model that is executed on target hardware
        compiledmodelpath : Path
            Path to the file with a compiled model
        """
        self.prepare_client()
        self.protocol.upload_model(compiledmodelpath)
        measurements = Measurements()
        try:
            for X, y in tqdm(iter(dataset)):
                prepX = modelwrapper._preprocess_input(X)
                prepX = modelwrapper.convert_input_to_bytes(prepX)
                check_request(self.protocol.upload_input(prepX), 'send input')
                check_request(self.protocol.request_processing(), 'inference')
                _, preds = check_request(
                    self.protocol.download_output(),
                    'receive output'
                )
                self.protocol.log.debug(
                    f'Received output ({len(preds)} bytes)'
                )
                preds = modelwrapper.convert_output_from_bytes(preds)
                posty = modelwrapper._postprocess_outputs(preds)
                measurements += dataset.evaluate(posty, y)

            measurements += self.protocol.download_statistics()
        except RequestFailure as ex:
            self.protocol.log.fatal(ex)
        else:
            MeasurementsCollector.measurements += measurements
        self.protocol.disconnect()

    def run_server(self):
        """
        Main runtime server program.

        It waits for requests from a single client.

        Based on requests, it loads the model, runs inference and provides
        statistics.
        """
        self.prepare_server()
        self.shouldwork = True
        while self.shouldwork:
            actions = self.protocol.wait_for_activity()
            for status, data in actions:
                if status == ServerStatus.DATA_READY:
                    if len(data) != 1:
                        self.protocol.log.error('Too many messages')
                        self.close_server()
                        self.shouldwork = False
                    msgtype, content = self.protocol.parse_message(data[0])
                    self.callbacks[msgtype](content)
                elif status == ServerStatus.DATA_INVALID:
                    self.protocol.log.error('Invalid message received')
                    self.protocol.log.error('Client will be disconnected')
                    self.disconnect()
        self.disconnect()
