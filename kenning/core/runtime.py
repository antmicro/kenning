"""
Module providing a Runtime wrapper.

Runtimes implement running and testing deployed models on target devices.
"""

import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Optional
import json

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.runtimeprotocol import RuntimeProtocol
from kenning.core.runtimeprotocol import MessageType
from kenning.core.runtimeprotocol import RequestFailure
from kenning.core.runtimeprotocol import check_request
from kenning.core.measurements import Measurements
from kenning.core.measurements import MeasurementsCollector
from kenning.core.runtimeprotocol import ServerStatus
from kenning.core.measurements import timemeasurements
from kenning.core.measurements import SystemStatsCollector
from kenning.utils.logger import get_logger
from kenning.core.measurements import systemstatsmeasurements


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
            MessageType.DATA: self._prepare_input,
            MessageType.MODEL: self._prepare_model,
            MessageType.PROCESS: self.process_input,
            MessageType.OUTPUT: self._upload_output,
            MessageType.STATS: self._upload_stats
        }
        self.statsmeasurements = None
        self.log = get_logger()

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
        protocol : RuntimeProtocol
            RuntimeProtocol object
        args : Dict
            arguments from ArgumentParser object

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

        Returns
        -------
        bool : True if succeded
        """
        raise NotImplementedError

    def _prepare_input(self, input_data: bytes):
        if self.prepare_input(input_data):
            self.protocol.request_success()
        else:
            self.protocol.request_failure()

    def _prepare_model(self, input_data: Optional[bytes]):
        """
        Internal call for preparing a model for inference task.

        Parameters
        ----------
        input_data : Optional[bytes]
            Model data or None, if the model should be loaded from another
            source.

        Returns
        -------
        bool : True if succeded
        """
        self.inference_session_start()
        ret = self.prepare_model(input_data)
        if ret:
            self.protocol.request_success()
        else:
            self.protocol.request_failure()
        return ret

    def prepare_model(self, input_data: Optional[bytes]) -> bool:
        """
        Receives the model to infer from the client in bytes.

        The method should load bytes with the model, optionally save to file
        and allocate the model on target device for inference.

        ``input_data`` stores the model representation in bytes.
        If ``input_data`` is None, the model is extracted from another source
        (i.e. from existing file).

        Parameters
        ----------
        input_data : Optional[bytes]
            Model data or None, if the model should be loaded from another
            source.

        Returns
        -------
        bool : True if succeded
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
        self.log.debug('Processing input')
        self.protocol.request_success()
        self._run()
        self.protocol.request_success()
        self.log.debug('Input processed')

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

    def upload_output(self, input_data: bytes) -> bytes:
        """
        Returns the output to the client, in bytes.

        The method converts the direct output from the model to bytes and
        returns them.

        The wrapper later sends the data to the client.

        Parameters
        ----------
        input_data : bytes
            Not used here

        Returns
        -------
        bytes : data to send to the client
        """
        raise NotImplementedError

    def _upload_output(self, input_data: bytes) -> bytes:
        out = self.upload_output(input_data)
        if out:
            self.protocol.request_success(out)
        else:
            self.protocol.request_failure()

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
        out = self.upload_stats(input_data)
        self.protocol.request_success(out)

    def upload_stats(self, input_data: bytes) -> bytes:
        """
        Returns statistics of inference passes to the client.

        Default implementation converts collected metrics in
        MeasurementsCollector to JSON format and returns them for sending.

        Parameters
        ----------
        input_data : bytes
            Not used here

        Returns
        -------
        bytes : statistics to be sent to the client
        """
        self.log.debug('Uploading stats')
        stats = json.dumps(MeasurementsCollector.measurements.data)
        return stats.encode('utf-8')

    @systemstatsmeasurements('full_run_statistics')
    def run_locally(
            self,
            dataset: Dataset,
            modelwrapper: ModelWrapper,
            compiledmodelpath: Path):
        """
        Runs inference locally using a given runtime.

        Parameters
        ----------
        dataset : Dataset
            Dataset to verify the inference on
        modelwrapper : ModelWrapper
            Model that is executed on target hardware
        compiledmodelpath : Path
            Path to the file with a compiled model

        Returns
        -------
        bool : True if executed successfully
        """
        measurements = Measurements()
        try:
            self.inference_session_start()
            self.prepare_model(None)
            for X, y in tqdm(iter(dataset)):
                prepX = modelwrapper._preprocess_input(X)
                prepX = modelwrapper.convert_input_to_bytes(prepX)
                self.prepare_input(prepX)
                self._run()
                outbytes = self.upload_output(None)
                preds = modelwrapper.convert_output_from_bytes(outbytes)
                posty = modelwrapper._postprocess_outputs(preds)
                measurements += dataset.evaluate(posty, y)
        finally:
            self.inference_session_end()
            MeasurementsCollector.measurements += measurements
        return True

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

        Returns
        -------
        bool : True if executed successfully
        """
        if self.protocol is None:
            raise RequestFailure('Protocol is not provided')
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
                self.log.debug(
                    f'Received output ({len(preds)} bytes)'
                )
                preds = modelwrapper.convert_output_from_bytes(preds)
                posty = modelwrapper._postprocess_outputs(preds)
                measurements += dataset.evaluate(posty, y)

            measurements += self.protocol.download_statistics()
        except RequestFailure as ex:
            self.log.fatal(ex)
            return False
        else:
            MeasurementsCollector.measurements += measurements
        self.protocol.disconnect()
        return True

    def run_server(self):
        """
        Main runtime server program.

        It waits for requests from a single client.

        Based on requests, it loads the model, runs inference and provides
        statistics.
        """
        if self.protocol is None:
            raise RequestFailure('Protocol is not provided')
        self.prepare_server()
        self.shouldwork = True
        while self.shouldwork:
            actions = self.protocol.wait_for_activity()
            for status, data in actions:
                if status == ServerStatus.DATA_READY:
                    if len(data) != 1:
                        self.log.error('Too many messages')
                        self.close_server()
                        self.shouldwork = False
                    msgtype, content = self.protocol.parse_message(data[0])
                    self.callbacks[msgtype](content)
                elif status == ServerStatus.DATA_INVALID:
                    self.log.error('Invalid message received')
                    self.log.error('Client will be disconnected')
                    self.disconnect()
        self.protocol.disconnect()
