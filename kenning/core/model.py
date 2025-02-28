# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a wrapper for deep learning models.
"""

import json
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Any, Dict, List, Optional, Tuple, Type
from urllib.request import HTTPError

import numpy as np

from kenning.core.dataset import Dataset
from kenning.core.measurements import (
    Measurements,
    MeasurementsCollector,
    systemstatsmeasurements,
    tagmeasurements,
    timemeasurements,
)
from kenning.interfaces.io_interface import IOInterface
from kenning.utils.args_manager import ArgumentsHandler, get_parsed_json_dict
from kenning.utils.logger import LoggerProgressBar, TqdmCallback
from kenning.utils.resource_manager import PathOrURI, ResourceURI


class VariableBatchSizeNotSupportedError(Exception):
    """
    Exception raised when trying to create a model which is not fitted to
    handle variable batch sizes yet.
    """

    def __init__(
        self,
        msg="Inference batch size greater than one not supported for this model.",  # noqa: E501
        *args,
        **kwargs,
    ):
        super().__init__(msg, *args, **kwargs)


class TrainingParametersMissingError(Exception):
    """
    Exception raised when trying train a model without defined training
    parameters.
    """

    def __init__(
        self,
        params,
        msg="Missing train parameters: {}",
        *args,
        **kwargs,
    ):
        super().__init__(msg.format(", ".join(params)), *args, **kwargs)


class ModelWrapper(IOInterface, ArgumentsHandler, ABC):
    """
    Wraps the given model.
    """

    pretrained_model_uri: Optional[str] = None
    default_dataset: Optional[Type[Dataset]] = None
    arguments_structure = {
        "model_path": {
            "argparse_name": "--model-path",
            "description": "Path to the model",
            "type": ResourceURI,
            "required": True,
        },
        "model_name": {
            "argparse_name": "--model-name",
            "description": "Name of the model used for the report",
            "type": str,
        },
    }

    def __init__(
        self,
        model_path: PathOrURI,
        dataset: Optional[Dataset],
        from_file: bool = True,
        model_name: Optional[str] = None,
    ):
        """
        Creates the model wrapper.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        dataset : Optional[Dataset]
            The dataset to verify the inference.
        from_file : bool
            True if the model should be loaded from file.
        model_name : Optional[str]
            Name of the model used for the report
        """
        self.model_path = model_path
        self.model_name = model_name
        self.dataset = dataset
        self.from_file = from_file
        self.model_prepared = False
        self.should_cancel = False

    def get_path(self) -> PathOrURI:
        """
        Returns path to the model in a form of a Path or ResourceURI object.

        Returns
        -------
        PathOrURI
            Path or URI to the model.
        """
        return self.model_path

    @classmethod
    def from_argparse(
        cls,
        dataset: Optional[Dataset],
        args: Namespace,
        from_file: bool = True,
    ) -> "ModelWrapper":
        """
        Constructor wrapper that takes the parameters from argparse args.

        Parameters
        ----------
        dataset : Optional[Dataset]
            The dataset object to feed to the model.
        args : Namespace
            Arguments from ArgumentParser object.
        from_file : bool
            Determines if the model should be loaded from model_path.

        Returns
        -------
        ModelWrapper
            Object of class ModelWrapper.
        """
        return super().from_argparse(
            args, dataset=dataset, from_file=from_file
        )

    @classmethod
    def from_json(
        cls,
        json_dict: Dict,
        dataset: Optional[Dataset] = None,
        from_file: bool = True,
    ) -> "ModelWrapper":
        """
        Constructor wrapper that takes the parameters from json dict.

        This function checks if the given dictionary is valid according
        to the ``arguments_structure`` defined.
        If it is then it invokes the constructor.

        Parameters
        ----------
        json_dict : Dict
            Arguments for the constructor.
        dataset : Optional[Dataset]
            The dataset object to feed to the model.
        from_file : bool
            Determines if the model should be loaded from model_path.

        Returns
        -------
        ModelWrapper
            Object of class ModelWrapper.
        """
        return super().from_json(
            json_dict, dataset=dataset, from_file=from_file
        )

    @abstractmethod
    def prepare_model(self):
        """
        Downloads the model (if required) and loads it to the device.

        Should be used whenever the model is actually required.

        The prepare_model method should check model_prepared field
        to determine if the model is not already loaded.

        It should also set model_prepared field to True
        once the model is prepared.
        """
        ...

    @abstractmethod
    def load_model(self, model_path: PathOrURI):
        """
        Loads the model from file.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        """
        ...

    @abstractmethod
    def save_model(self, model_path: PathOrURI):
        """
        Saves the model to file.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        """
        ...

    @abstractmethod
    def save_to_onnx(self, model_path: PathOrURI):
        """
        Saves the model in the ONNX format.

        Parameters
        ----------
        model_path : PathOrURI
            Path or URI to the model file.
        """
        ...

    def preprocess_input(self, X: List[Any]) -> List[Any]:
        """
        Preprocesses the inputs for a given model before inference.

        By default no action is taken, and the inputs are passed unmodified.

        Parameters
        ----------
        X : List[Any]
            The input data from the Dataset object.

        Returns
        -------
        List[Any]
            The preprocessed inputs that are ready to be fed to the model.
        """
        return X

    def _quantize_inputs(
        self,
        X: List[Any],
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ) -> List[Any]:
        """
        Reshapes input data to the format expected by the model.
        Applies quantization if needed.

        Parameters
        ----------
        X : List[Any]
            Input data to be preprocessed.
        io_spec : Optional[Dict[str, List[Dict]]]
            IO specification to be used. If not specified, then it is
            retrieved from model.

        Returns
        -------
        List[Any]
            List of preprocessed input data.
        """
        if io_spec is None:
            io_spec = self.get_io_specification()

        for idx, (io_spec, inp) in enumerate(
            zip(
                io_spec.get("processed_input", io_spec["input"]),
                X,
            )
        ):
            # Check if dtype is valid and if it should be quantized
            if "prequantized_dtype" in io_spec and inp.dtype != np.dtype(
                io_spec["dtype"]
            ):
                scale = io_spec["scale"]
                zero_point = io_spec["zero_point"]
                X[idx] = (inp / scale + zero_point).astype(io_spec["dtype"])

        return X

    def _preprocess_input(
        self,
        X: List[Any],
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ) -> List[Any]:
        if io_spec is None:
            io_spec = self.get_io_specification()

        IOInterface.assert_data_format(X, io_spec["input"])

        preprocessed_x = timemeasurements("input_preprocess_step")(
            tagmeasurements("preprocess")(self.preprocess_input)
        )(X)

        preprocessed_x = self._quantize_inputs(preprocessed_x, io_spec)

        IOInterface.assert_data_format(
            preprocessed_x,
            io_spec[
                "processed_input" if "processed_input" in io_spec else "input"
            ],
        )
        return preprocessed_x

    def postprocess_outputs(self, y: List[Any]) -> List[Any]:
        """
        Processes the outputs for a given model.

        By default no action is taken, and the outputs are passed unmodified.

        Parameters
        ----------
        y : List[Any]
            The list of output data from the model.

        Returns
        -------
        List[Any]
            The post processed outputs from the model that need to be in
            format requested by the Dataset object.
        """
        return y

    def _dequantize_outputs(
        self,
        y: List[Any],
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ) -> List[Any]:
        """
        The method accepts output of the model and postprocesses it.

        The output is quantized and converted to a correct dtype if needed.

        Some compilers can change the order of the layers. If that's the case
        the methods also reorders the output to match the original
        order of the model before compilation.

        Parameters
        ----------
        y : List[Any]
            List of outputs of the model.
        io_spec : Optional[Dict[str, List[Dict]]]
            IO specification to be used. If not specified, then it is
            retrieved from model.

        Returns
        -------
        List[Any]
            Postprocessed and reordered outputs of the model.
        """
        if io_spec is None:
            io_spec = self.get_io_specification()

        is_reordered = any(["order" in spec for spec in io_spec["output"]])

        # dequantization/precision conversion
        for i, output_spec in enumerate(io_spec["output"]):
            if "prequantized_dtype" in output_spec:
                if ("scale" not in output_spec) and (
                    "zero_point" not in output_spec
                ):
                    y[i] = y[i].astype(output_spec["prequantized_dtype"])
                else:
                    scale = output_spec.get("scale", 1.0)
                    zero_point = output_spec.get("zero_point", 0.0)
                    y[i] = (
                        y[i].astype(output_spec["prequantized_dtype"])
                        - zero_point
                    ) * scale

        # retrieving original order
        reordered_results = [None] * len(y)
        if is_reordered:
            for output_spec, result in zip(io_spec["output"], y):
                reordered_results[output_spec["order"]] = result
        else:
            reordered_results = y

        return reordered_results

    def _postprocess_outputs(
        self,
        y: List[Any],
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ) -> List[Any]:
        if io_spec is None:
            io_spec = self.get_io_specification()

        IOInterface.assert_data_format(y, io_spec["output"])

        processed_y = self._dequantize_outputs(y, io_spec)

        processed_y = timemeasurements("output_postprocess_step")(
            tagmeasurements("postprocess")(self.postprocess_outputs)
        )(processed_y)

        IOInterface.assert_data_format(
            processed_y,
            io_spec.get("processed_output", io_spec["output"]),
        )
        return processed_y

    @abstractmethod
    def run_inference(self, X: List[Any]) -> List[Any]:
        """
        Runs inference for a given preprocessed input.

        Parameters
        ----------
        X : List[Any]
            The preprocessed inputs for the model.

        Returns
        -------
        List[Any]
            The results of the inference.
        """
        ...

    @abstractmethod
    def get_framework_and_version(self) -> Tuple[str, str]:
        """
        Returns name of the framework and its version in a form of a tuple.

        Returns
        -------
        Tuple[str, str]
            Framework name and version.
        """
        ...

    @abstractmethod
    def get_output_formats(self) -> List[str]:
        """
        Returns list of names of possible output formats.

        Returns
        -------
        List[str]
            List of possible output format names.
        """
        ...

    @timemeasurements("target_inference_step")
    def _run_inference(self, X):
        return self.run_inference(X)

    @systemstatsmeasurements("session_utilization")
    @timemeasurements("inference")
    def test_inference(self) -> "Measurements":
        """
        Runs the inference on test split of the dataset.

        Returns
        -------
        Measurements
            The inference results.
        """
        measurements = Measurements()

        with LoggerProgressBar() as logger_progress_bar:
            for X, y in TqdmCallback(
                "runtime", self.dataset.iter_test(), file=logger_progress_bar
            ):
                if self.should_cancel:
                    break
                prepX = self._preprocess_input(X)
                preds = self._run_inference(prepX)
                posty = self._postprocess_outputs(preds)
                measurements += self.dataset._evaluate(
                    posty,
                    y,
                    self.io_specification["processed_output"]
                    if "processed_output" in self.io_specification
                    else self.io_specification["output"],
                )

        MeasurementsCollector.measurements += measurements

        return measurements

    def train_model(self):
        """
        Trains the model with a given dataset.

        This method should implement training routine for a given dataset and
        save a working model to a given path in a form of a single file.

        The training should be performed with given batch size, learning rate,
        and number of epochs.

        The model needs to be saved explicitly.

        Raises
        ------
        NotImplementedError
            Raised when method is not implemented
        """
        raise NotImplementedError

    @abstractmethod
    def get_io_specification_from_model(self) -> Dict[str, List[Dict]]:
        """
        Returns a new instance of dictionary with `input` and `output`
        keys that map to input and output specifications.

        A single specification is a list of dictionaries with
        names, shapes and dtypes for each layer. The order of the
        dictionaries is assumed to be expected by the `ModelWrapper`.

        It is later used in optimization and compilation steps.

        It is used by `get_io_specification` function to get the
        specification and save it for later use.

        Returns
        -------
        Dict[str, List[Dict]]
            Dictionary that conveys input and output
            layers specification.
        """
        ...

    def get_io_specification(self) -> Dict[str, List[Dict]]:
        """
        Returns a saved dictionary with `input` and `output` keys
        that map to input and output specifications.

        A single specification is a list of dictionaries with
        names, shapes and dtypes for each layer. The order of the
        dictionaries is assumed to be expected by the `ModelWrapper`.

        It is later used in optimization and compilation steps.

        Returns
        -------
        Dict[str, List[Dict]]
            Dictionary that conveys input and output
            layers specification.
        """
        if not hasattr(self, "io_specification"):
            self.io_specification = self.get_io_specification_from_model()
        return self.io_specification

    @classmethod
    def parse_io_specification_from_json(cls, json_dict):
        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)
        model_path = ResourceURI(parsed_json_dict["model_path"])
        io_spec = model_path.with_suffix(model_path.suffix + ".json")
        try:
            with open(io_spec, "r") as f:
                return json.load(f)
        except (FileNotFoundError, HTTPError):
            return cls.derive_io_spec_from_json_params(parsed_json_dict)

    @classmethod
    @abstractmethod
    def derive_io_spec_from_json_params(
        cls, json_dict: Dict
    ) -> Dict[str, List[Dict]]:
        """
        Creates IO specification by deriving parameters from parsed JSON
        dictionary. The resulting IO specification may differ from the results
        of `get_io_specification`, information that couldn't be retrieved from
        JSON parameters are absent from final IO spec or are filled with
        general value (example: '-1' for unknown dimension shape).

        Parameters
        ----------
        json_dict : Dict
            JSON dictionary formed by parsing the input JSON with
            ModelWrapper's parameterschema.

        Returns
        -------
        Dict[str, List[Dict]]
            Dictionary that conveys input and output
            layers specification.
        """
        ...

    @abstractmethod
    def convert_input_to_bytes(self, inputdata: Any) -> bytes:
        """
        Converts the input returned by the ``preprocess_input`` method
        to bytes.

        Parameters
        ----------
        inputdata : Any
            The preprocessed inputs.

        Returns
        -------
        bytes
            Input data as byte stream.
        """
        ...

    @abstractmethod
    def convert_output_from_bytes(self, outputdata: bytes) -> List[Any]:
        """
        Converts bytes array to the model output format.

        The converted output should be compatible with ``postprocess_outputs``
        method.

        Parameters
        ----------
        outputdata : bytes
            Output data in raw bytes.

        Returns
        -------
        List[Any]
            List of output data from a model. The converted data should be
            compatible with the ``postprocess_outputs`` method.
        """
        ...
