# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module providing a Runtime wrapper.

Runtimes implement running and testing deployed models on target devices.
"""

import json
import time
from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np

from kenning.core.measurements import (
    MeasurementsCollector,
    SystemStatsCollector,
    tagmeasurements,
    timemeasurements,
)
from kenning.core.model import ModelWrapper
from kenning.interfaces.io_interface import IOSpecWrongFormat
from kenning.utils.args_manager import ArgumentsHandler
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI


class ModelNotPreparedError(Exception):
    """
    Exception raised when trying to run the model without loading it first.
    """

    def __init__(
        self,
        msg="Make sure to run prepare_model method before running it.",
        *args,
        **kwargs,
    ):
        super().__init__(msg, *args, **kwargs)


class InputNotPreparedError(Exception):
    """
    Exception raised when trying to run the model without loading the inputs
    first.
    """

    def __init__(
        self,
        msg="Make sure to run load_input method before running the model.",
        *args,
        **kwargs,
    ):
        super().__init__(msg, *args, **kwargs)


class Runtime(ArgumentsHandler, ABC):
    """
    Runtime object provides an API for testing inference on target devices.
    """

    inputtypes = []

    arguments_structure = {
        "disable_performance_measurements": {
            "argparse_name": "--disable-performance-measurements",
            "description": "Disable collection and processing of performance "
            "metrics",
            "type": bool,
            "default": False,
        }
    }

    def __init__(self, disable_performance_measurements: bool = False):
        """
        Creates Runtime object.

        Parameters
        ----------
        disable_performance_measurements : bool
            Disable collection and processing of performance metrics.
        """
        self.statsmeasurements = None
        self.disable_performance_measurements = (
            disable_performance_measurements
        )
        self.input_spec = None
        self.output_spec = None
        self.processed_input_spec = None
        self.processed_output_spec = None
        self.misc_io_metadata = None

    @classmethod
    def from_argparse(cls, args: Namespace) -> "Runtime":
        """
        Constructor wrapper that takes the parameters from argparse args.

        Parameters
        ----------
        args : Namespace
            Arguments from ArgumentParser object.

        Returns
        -------
        Runtime
            Object of class Runtime.
        """
        return super().from_argparse(args)

    @classmethod
    def from_json(cls, json_dict: Dict) -> "Runtime":
        """
        Constructor wrapper that takes the parameters from json dict.

        This function checks if the given dictionary is valid according
        to the ``arguments_structure`` defined.
        If it is then it invokes the constructor.

        Parameters
        ----------
        json_dict : Dict
            Arguments for the constructor.

        Returns
        -------
        Runtime
            Object of class Runtime.
        """
        return super().from_json(json_dict)

    @classmethod
    def get_input_formats(cls) -> List[str]:
        """
        Returns list of names of possible input formats names.

        Returns
        -------
        List[str]
            List of possible input format names.
        """
        return cls.inputtypes

    def inference_session_start(self):
        """
        Calling this function indicates that the client is connected.

        This method should be called once the client has connected to a server.

        This will enable performance tracking.
        """
        if not self.disable_performance_measurements:
            if self.statsmeasurements is None:
                self.statsmeasurements = SystemStatsCollector(
                    "session_utilization"
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
            MeasurementsCollector.measurements += (
                self.statsmeasurements.get_measurements()
            )
            self.statsmeasurements = None

    @abstractmethod
    def load_input(self, input_data: List[Any]) -> bool:
        """
        Loads and converts delivered data to the accelerator for inference.

        This method is called when the input is received from the client.
        It is supposed to prepare input before running inference.

        Parameters
        ----------
        input_data : List[Any]
            Input data in bytes delivered by the client, preprocessed.

        Returns
        -------
        bool
            True if succeeded.

        Raises
        ------
        ModelNotLoadedError :
            Raised if model is not loaded.
        """
        ...

    @abstractmethod
    def prepare_model(self, input_data: Optional[bytes]) -> bool:
        """
        Receives the model to infer from the client in bytes.

        The method should load bytes with the model, optionally save to file
        and allocate the model on target device for inference.

        ``input_data`` stores the model representation in bytes.
        If ``input_data`` is None, the model is extracted from another source
        (i.e. from existing file or directory).

        Parameters
        ----------
        input_data : Optional[bytes]
            Model data or None, if the model should be loaded from another
            source.

        Returns
        -------
        bool
            True if succeeded.
        """
        ...

    def load_input_from_bytes(self, input_data: bytes) -> bool:
        """
        The method accepts `input_data` in bytes and loads it
        according to the input specification.

        It creates `np.ndarray` for every input layer using the metadata
        in `self.input_spec` and quantizes the data if needed.

        Some compilers can change the order of the layers. If that's the case
        the method also reorders the layers to match
        the specification of the model.

        Parameters
        ----------
        input_data : bytes
            Input data in bytes delivered by the client.

        Returns
        -------
        bool
            Output of the `load_input` method indicating if the operation
            succeeded.

        Raises
        ------
        AttributeError
            Raised if output specification is not loaded.
        """
        input_spec = (
            self.processed_input_spec
            if self.processed_input_spec
            else self.input_spec
        )
        if input_spec is None:
            raise AttributeError(
                "You must load the input specification first."
            )

        is_reordered = any(["order" in spec for spec in input_spec])
        if is_reordered:
            reordered_input_spec = sorted(
                input_spec, key=lambda spec: spec["order"]
            )
        else:
            reordered_input_spec = input_spec

        if not input_data:
            KLogger.error("Received empty data payload")
            return False

        # reading input
        inputs = []
        for spec in reordered_input_spec:
            shape = spec["shape"]
            # get original model dtype
            dtype = spec.get("prequantized_dtype", spec["dtype"])

            expected_size = np.abs(np.prod(shape) * np.dtype(dtype).itemsize)

            if len(input_data) % (expected_size / shape[0]) != 0:
                KLogger.error(
                    "Received input data that is not a multiple of the sample "
                    "size"
                )
                return False

            input = np.frombuffer(input_data[:expected_size], dtype=dtype)
            input = input.reshape(shape)

            inputs.append(input)
            input_data = input_data[expected_size:]

        if input_data:
            KLogger.error("Received more data than model expected")
            return False

        # retrieving original order
        if is_reordered:
            reordered_inputs = [None] * len(inputs)
            for order, spec in enumerate(input_spec):
                reordered_inputs[order] = inputs[spec["order"]]
        else:
            reordered_inputs = inputs

        return self.load_input(reordered_inputs)

    def preprocess_model_to_upload(self, path: PathOrURI) -> PathOrURI:
        """
        The method preprocesses the model to be uploaded to the client and
        returns a new path to it.

        The method is used to prepare the model to be sent to the client.
        It can be used to change the model representation, for example,
        to compress it.

        Parameters
        ----------
        path : PathOrURI
            Path to the model to preprocess.

        Returns
        -------
        PathOrURI
            Path to the preprocessed model.
        """
        return path

    def read_io_specification(self, io_spec: Dict):
        """
        Saves input/output specification so that it can be used during
        the inference.

        `input_spec` and `output_spec` are lists, where every
        element is a dictionary mapping (property name) -> (property value)
        for the layers.

        The standard property names are: `name`, `dtype` and `shape`.

        If the model is quantized it also has `scale`, `zero_point` and
        `prequantized_dtype` properties.

        If the layers of the model are reorder it also has `order` property.

        Parameters
        ----------
        io_spec : Dict
            Specification of the input/output layers.

        Raises
        ------
        IOSpecWrongFormat
            Raised if preprocessed input data has more
            than one available shape.
        """
        self.input_spec = io_spec["input"]
        self.output_spec = io_spec["output"]
        self.processed_input_spec = (
            io_spec["processed_input"]
            if "processed_input" in io_spec
            else None
        )
        self.processed_output_spec = (
            io_spec["processed_output"]
            if "processed_output" in io_spec
            else None
        )
        self.misc_io_metadata = {
            k: v
            for k, v in io_spec.items()
            if k
            not in ["input", "output", "processed_input", "processed_output"]
        }

        # Check if model input specification contains only one possible shape
        if any(
            "shape" in spec and isinstance(spec["shape"][0], Iterable)
            for spec in (
                self.processed_input_spec
                if self.processed_input_spec
                else self.input_spec
            )
        ):
            raise IOSpecWrongFormat(
                "Specification of input data after preprocessing has to have only one possible shape"  # noqa: E501
            )

    def prepare_io_specification(self, input_data: Optional[bytes]) -> bool:
        """
        Receives the io_specification from the client in bytes and saves
        it for later use.

        ``input_data`` stores the io_specification representation in bytes.
        If ``input_data`` is None, the io_specification is extracted
        from another source (i.e. from existing file). If it can not be
        found in this path, io_specification is not loaded.

        When no specification file is found, the function returns True as some
        Runtimes may not need io_specification to run the inference.

        Parameters
        ----------
        input_data : Optional[bytes]
            The io_specification` or None, if it should be loaded
            from another source.

        Returns
        -------
        bool
            True if succeeded.
        """
        if input_data is None:
            path = self.get_io_spec_path(self.model_path)
            if not path.exists():
                KLogger.info("No Input/Output specification found")
                return False

            with open(path, "rb") as f:
                try:
                    io_spec = json.load(f)
                except json.JSONDecodeError as e:
                    KLogger.warning(
                        f"Error while parsing IO specification: {e}"
                    )
                    return False
        else:
            io_spec = json.loads(input_data)

        self.read_io_specification(io_spec)
        KLogger.info("Input/Output specification loaded")
        return True

    def get_io_spec_path(self, model_path: PathOrURI) -> Path:
        """
        Gets path to a input/output specification file which is
        `model_path` and `.json` concatenated.

        Parameters
        ----------
        model_path : PathOrURI
            URI to the compiled model.

        Returns
        -------
        Path
            Returns path to the specification.
        """
        spec_path = model_path.with_suffix(model_path.suffix + ".json")

        return Path(str(spec_path))

    @timemeasurements("target_inference_step")
    @tagmeasurements("inference")
    def _run(self):
        """
        Performance wrapper for run method.
        """
        self.run()

    @abstractmethod
    def run(self):
        """
        Runs inference on prepared input.

        The input should be introduced in runtime's model representation, or
        it should be delivered using a variable that was assigned in
        load_input method.

        Raises
        ------
        ModelNotLoadedError :
            Raised if model is not loaded.
        """
        ...

    @abstractmethod
    def extract_output(self) -> List[Any]:
        """
        Extracts and postprocesses the output of the model.

        Returns
        -------
        List[Any]
            Postprocessed and reordered outputs of the model.
        """
        ...

    def upload_output(self, input_data: bytes) -> bytes:
        """
        Returns the output to the client, in bytes.

        The method converts the direct output from the model to bytes and
        returns them.

        The wrapper later sends the data to the client.

        Parameters
        ----------
        input_data : bytes
            Not used here.

        Returns
        -------
        bytes
            Data to send to the client.
        """
        KLogger.debug("Uploading output")
        results: List[Union(np.ndarray, str)] = self.extract_output()
        output_bytes = bytes()
        for result in results:
            if isinstance(result, str):
                # Length is added to the beginning of the string
                # to allow proper decoding
                output_bytes += f"{len(result)} {result}".encode("utf-8")
            else:
                output_bytes += result.tobytes()
        return output_bytes

    def upload_stats(self, input_data: bytes) -> bytes:
        """
        Returns statistics of inference passes to the client.

        Default implementation converts collected metrics in
        MeasurementsCollector to JSON format and returns them for sending.

        Parameters
        ----------
        input_data : bytes
            Not used here.

        Returns
        -------
        bytes
            Statistics to be sent to the client.
        """
        KLogger.debug("Uploading stats")
        stats = json.dumps(MeasurementsCollector.measurements.data)
        return stats.encode("utf-8")

    def prepare_local(self) -> bool:
        """
        Runs initialization for the local inference.

        Returns
        -------
        bool
            True if initialized successfully.
        """
        return self.prepare_io_specification(None) and self.prepare_model(None)

    def infer(
        self,
        X: List[np.ndarray],
        model_wrapper: ModelWrapper,
        postprocess: bool = True,
    ) -> List[Any]:
        """
        Runs inference on single batch locally using a given runtime.

        Parameters
        ----------
        X : List[np.ndarray]
            Batch of data provided for inference.
        model_wrapper : ModelWrapper
            Model that is executed on target hardware.
        postprocess : bool
            Indicates if model output should be postprocessed.

        Returns
        -------
        List[Any]
            Obtained values.
        """
        prepX = model_wrapper._preprocess_input(X)
        succeed = self.load_input(prepX)
        if not succeed:
            return False
        self._run()
        preds = self.extract_output()
        if postprocess:
            return model_wrapper._postprocess_outputs(preds)

        return preds

    def get_time(self) -> float:
        """
        Gets the current timestamp.

        Returns
        -------
        float
            Current timestamp.
        """
        return time.perf_counter()
