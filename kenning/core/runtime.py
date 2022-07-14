"""
Module providing a Runtime wrapper.

Runtimes implement running and testing deployed models on target devices.
"""

import argparse
from pathlib import Path
from typing import Optional, Dict
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
from kenning.core.measurements import tagmeasurements
from kenning.core.measurements import SystemStatsCollector
from kenning.utils.logger import get_logger
from kenning.core.measurements import systemstatsmeasurements
from kenning.utils.args_manager import add_parameterschema_argument, add_argparse_argument, get_parsed_json_dict  # noqa: E501


class Runtime(object):
    """
    Runtime object provides an API for testing inference on target devices.

    Using a provided RuntimeProtocol it sets up a client (host) and server
    (target) communication, during which the inference metrics are being
    analyzed.
    """

    arguments_structure = {
        'collect_performance_data': {
            'argparse_name': '--disable-performance-measurements',
            'description': 'Disable collection and processing of performance metrics',  # noqa: E501
            'type': bool,
            'default': True
        }
    }

    def __init__(
            self,
            protocol: RuntimeProtocol,
            collect_performance_data: bool = True):
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
        self.collect_performance_data = collect_performance_data

    @classmethod
    def _form_argparse(cls):
        """
        Wrapper for creating argparse structure for the Runtime class.

        Returns
        -------
        ArgumentParser :
            the argument parser object that can act as parent for program's
            argument parser
        """
        parser = argparse.ArgumentParser(add_help=False)
        group = parser.add_argument_group(title='Runtime arguments')
        add_argparse_argument(
            group,
            Runtime.arguments_structure
        )
        return parser, group

    @classmethod
    def form_argparse(cls):
        """
        Creates argparse parser for the Runtime object.

        Returns
        -------
        ArgumentParser :
            the argument parser object that can act as parent for program's
            argument parser
        """
        parser, group = cls._form_argparse()
        if cls.arguments_structure != Runtime.arguments_structure:
            add_argparse_argument(
                group,
                cls.arguments_structure
            )
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
        RuntimeProtocol : object of class RuntimeProtocol
        """
        return cls(protocol, args.disable_performance_measurements)

    @classmethod
    def _form_parameterschema(cls):
        """
        Wrapper for creating parameterschema structure for the Runtime class.

        Returns
        -------
        Dict : schema for the class
        """
        parameterschema = {
            "type": "object",
            "additionalProperties": False
        }

        add_parameterschema_argument(
            parameterschema,
            Runtime.arguments_structure,
        )

        return parameterschema

    @classmethod
    def form_parameterschema(cls):
        """
        Creates schema for the Runtime class.

        Returns
        -------
        Dict : schema for the class
        """
        parameterschema = cls._form_parameterschema()
        if cls.arguments_structure != Runtime.arguments_structure:
            add_parameterschema_argument(
                parameterschema,
                cls.arguments_structure
            )
        return parameterschema

    @classmethod
    def from_json(cls, protocol: RuntimeProtocol, json_dict: Dict):
        """
        Constructor wrapper that takes the parameters from json dict.

        This function checks if the given dictionary is valid according
        to the ``arguments_structure`` defined.
        If it is then it invokes the constructor.

        Parameters
        ----------
        protocol : RuntimeProtocol
            RuntimeProtocol object
        json_dict : Dict
            Arguments for the constructor

        Returns
        -------
        Runtime : object of class Runtime
        """

        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)

        return cls(
            protocol,
            **parsed_json_dict
        )

    def inference_session_start(self):
        """
        Calling this function indicates that the client is connected.

        This method should be called once the client has connected to a server.

        This will enable performance tracking.
        """
        if self.collect_performance_data:
            if self.statsmeasurements is None:
                self.statsmeasurements = SystemStatsCollector(
                    'session_utilization'
                )
            self.statsmeasurements.start()
        else:
            self.statsmeasurements = None

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
    @tagmeasurements('inference')
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

    def upload_essentials(self, compiledmodelpath: Path):
        """
        Wrapper for uploading data to the server.
        Uploads model by default.

        Parameters
        ----------
        compiledmodelpath : Path
            Path to the file with a compiled model

        """
        self.protocol.upload_model(compiledmodelpath)

    def prepare_local(self):
        """
        Runs initialization for the local inference.
        """
        self.prepare_model(None)

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
        from tqdm import tqdm
        measurements = Measurements()
        try:
            self.inference_session_start()
            self.prepare_local()
            for X, y in tqdm(iter(dataset)):
                prepX = tagmeasurements("preprocessing")(modelwrapper._preprocess_input)(X)  # noqa: 501
                prepX = modelwrapper.convert_input_to_bytes(prepX)
                succeed = self.prepare_input(prepX)
                if not succeed:
                    return False
                self._run()
                outbytes = self.upload_output(None)
                preds = modelwrapper.convert_output_from_bytes(outbytes)
                posty = tagmeasurements("postprocessing")(modelwrapper._postprocess_outputs)(preds)  # noqa: 501
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
        from tqdm import tqdm
        if self.protocol is None:
            raise RequestFailure('Protocol is not provided')
        self.prepare_client()
        self.upload_essentials(compiledmodelpath)
        measurements = Measurements()
        try:
            for X, y in tqdm(iter(dataset)):
                prepX = tagmeasurements("preprocessing")(modelwrapper._preprocess_input)(X)  # noqa: 501
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
                posty = tagmeasurements("postprocessing")(modelwrapper._postprocess_outputs)(preds)  # noqa: 501
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
