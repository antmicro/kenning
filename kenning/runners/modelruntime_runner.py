# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a runner that performs inference.
"""
from argparse import Namespace
from copy import deepcopy
from typing import Any, Dict, List, Tuple

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.runner import Runner
from kenning.core.runtime import Runtime
from kenning.utils.args_manager import (
    get_parsed_args_dict,
    get_parsed_json_dict,
)
from kenning.utils.class_loader import any_from_json, load_class


class ModelRuntimeRunner(Runner):
    """
    Runner that performs inference using given model and runtime.
    """

    arguments_structure = {
        "model_wrapper": {
            "argparse_name": "--model-wrapper",
            "description": "Path to JSON describing the ModelWrapper object, "
            "following its argument structure",
            "type": object,
            "required": True,
        },
        "dataset": {
            "argparse_name": "--dataset",
            "description": "Path to JSON describing the Dataset object, "
            "following its argument structure",
            "type": object,
        },
        "runtime": {
            "argparse_name": "--runtime",
            "description": "Path to JSON describing the Runtime object, "
            "following its argument structure",
            "type": object,
            "required": True,
        },
    }

    def __init__(
        self,
        model: ModelWrapper,
        runtime: Runtime,
        inputs_sources: Dict[str, Tuple[int, str]] = {},
        inputs_specs: Dict[str, Dict] = {},
        outputs: Dict[str, str] = {},
    ):
        """
        Creates the model runner.

        Parameters
        ----------
        model : ModelWrapper
            Selected model.
        runtime : Runtime
            Runtime used to run selected model.
        inputs_sources : Dict[str, Tuple[int, str]]
            Input from where data is being retrieved.
        inputs_specs : Dict[str, Dict]
            Specifications of runner's inputs.
        outputs : Dict[str, str]
            Outputs of this Runner.
        """
        self.model = model
        self.runtime = runtime

        self.runtime.inference_session_start()
        self.runtime.prepare_local()

        super().__init__(
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs,
        )

    def cleanup(self):
        self.runtime.inference_session_end()

    @classmethod
    def from_argparse(
        cls,
        args: Namespace,
        inputs_sources: Dict[str, Tuple[int, str]],
        inputs_specs: Dict[str, Dict],
        outputs: Dict[str, str],
    ) -> "ModelRuntimeRunner":
        parsed_json_dict = get_parsed_args_dict(cls, args)

        return cls._from_parsed_json_dict(
            parsed_json_dict=parsed_json_dict,
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs,
        )

    @classmethod
    def from_json(
        cls,
        json_dict: Dict,
        inputs_sources: Dict[str, Tuple[int, str]],
        inputs_specs: Dict[str, Dict],
        outputs: Dict[str, str],
    ) -> "ModelRuntimeRunner":
        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)

        return cls._from_parsed_json_dict(
            parsed_json_dict=parsed_json_dict,
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs,
        )

    @classmethod
    def _from_parsed_json_dict(
        cls,
        parsed_json_dict: Dict[str, Any],
        inputs_sources: Dict[str, Tuple[int, str]],
        inputs_specs: Dict[str, Dict],
        outputs: Dict[str, str],
    ) -> "ModelRuntimeRunner":
        if parsed_json_dict.get("dataset", None):
            dataset = cls._create_dataset(parsed_json_dict["dataset"])
        else:
            dataset = None

        model = cls._create_model(dataset, parsed_json_dict["model_wrapper"])
        model.prepare_model()

        runtime = cls._create_runtime(parsed_json_dict["runtime"])

        return cls(
            model,
            runtime,
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs,
        )

    @staticmethod
    def _create_dataset(json_dict: Dict) -> Dataset:
        """
        Method used to create dataset based on json dict.

        Parameters
        ----------
        json_dict : Dict
            Arguments for the constructor.

        Returns
        -------
        Dataset
            Created dataset.
        """
        return any_from_json(json_dict)

    @staticmethod
    def _create_model(dataset: Dataset, json_dict: Dict) -> ModelWrapper:
        """
        Method used to create model based on json dict.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to initialize model parameters (class names etc.).
        json_dict : Dict
            Arguments for the constructor.

        Returns
        -------
        ModelWrapper
            Created model.
        """
        return any_from_json(json_dict, dataset=dataset)

    @staticmethod
    def _create_runtime(json_dict: Dict) -> Runtime:
        """
        Method used to create runtime based on json dict.

        Parameters
        ----------
        json_dict : Dict
            Arguments for the constructor.

        Returns
        -------
        Runtime
            Created runtime.
        """
        return any_from_json(json_dict)

    @classmethod
    def _get_io_specification(
        cls, model_io_spec: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """
        Creates runner IO specification from chosen parameters.

        Parameters
        ----------
        model_io_spec : Dict[str, List[Dict]]
            Model IO specification.

        Returns
        -------
        Dict[str, List[Dict]]
            Dictionary that conveys input and output layers specification.
        """
        for io in ("input", "output"):
            if f"processed_{io}" not in model_io_spec.keys():
                model_io_spec[f"processed_{io}"] = []
                for spec in model_io_spec[io]:
                    spec = deepcopy(spec)
                    spec["name"] = "processed_" + spec["name"]
                    model_io_spec[f"processed_{io}"].append(spec)

        return model_io_spec

    @classmethod
    def parse_io_specification_from_json(cls, json_dict):
        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)

        model_json_dict = parsed_json_dict["model_wrapper"]
        model_cls = load_class(model_json_dict["type"])
        model_io_spec = model_cls.parse_io_specification_from_json(
            model_json_dict["parameters"]
        )
        return cls._get_io_specification(model_io_spec)

    def get_io_specification(self) -> Dict[str, List[Dict]]:
        return self._get_io_specification(self.model.get_io_specification())

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_input = inputs.get("processed_input")
        if model_input is None:
            model_input = inputs["input"]

        preds = self.runtime.infer(model_input, self.model, postprocess=False)
        posty = self.model.postprocess_outputs(preds)

        io_spec = self.get_io_specification()

        result = {}
        # TODO: Add support for multiple inputs/outputs
        for out_spec, out_value in zip(io_spec["output"], [preds]):
            result[out_spec["name"]] = out_value

        for out_spec, out_value in zip(io_spec["processed_output"], [posty]):
            result[out_spec["name"]] = out_value

        return result
