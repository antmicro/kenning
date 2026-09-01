# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides a runner that performs inference.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from kenning.core.exceptions import KenningError
from kenning.core.model import ModelWrapper
from kenning.core.runner import Runner
from kenning.core.runtime import Runtime
from kenning.dispatcher.block_config import (
    ConfigKey,
    yaml_or_json_to_config_dict,
)
from kenning.utils.args_manager import (
    get_parsed_json_dict,
)
from kenning.utils.class_loader import load_class, objs_from_full_dict_config


class ModelRuntimeRunner(Runner):
    """
    Runner that performs inference using given model and runtime.
    """

    arguments_structure = {
        "model_wrapper": {
            "argparse_name": "--model-wrapper",
            "description": "JSON describing the ModelWrapper object, "
            "following its argument structure",
            "type": object,
            "required": True,
        },
        "runtime": {
            "argparse_name": "--runtime",
            "description": "JSON describing the Runtime object, "
            "following its argument structure",
            "type": object,
            "required": True,
        },
        "dataset": {
            "argparse_name": "--dataset",
            "description": "JSON describing the Dataset object, "
            "following its argument structure",
            "type": object,
            "default": None,
        },
    }

    def __init__(
        self,
        model_wrapper: Dict[str, Any],
        runtime: Dict[str, Any],
        dataset: Optional[Dict[str, Any]],
        inputs_sources: Dict[str, Tuple[int, str]] = {},
        inputs_specs: Dict[str, Dict] = {},
        outputs: Dict[str, str] = {},
    ):
        """
        Creates the model runner.

        Parameters
        ----------
        model_wrapper : Dict[str, Any]
            JSON describing the ModelWrapper object, following its argument
            structure
        runtime : Dict[str, Any]
            JSON describing the Runtime object, following its argument
            structure.
        dataset: Optional[Dict[str, Any]]
            JSON describing the Dataset object, following its argument
            structure.
        inputs_sources : Dict[str, Tuple[int, str]]
            Input from where data is being retrieved.
        inputs_specs : Dict[str, Dict]
            Specifications of runner's inputs.
        outputs : Dict[str, str]
            Outputs of this Runner.
        """
        (
            self.model,
            self.runtime,
        ) = ModelRuntimeRunner._create_model_and_runtime(
            model_wrapper, runtime, dataset
        )
        self.runtime.inference_session_start()
        self.runtime.prepare_local()
        super().__init__(
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs,
        )

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

    def get_model_io_specification(self) -> Dict[str, List[Dict]]:
        return self._get_io_specification(self.model.get_io_specification())

    def get_io_specification(self) -> Dict[str, List[Dict]]:
        model_spec = self._get_io_specification(
            self.model.get_io_specification()
        )
        input_spec = []
        for spec in model_spec["input"]:
            spec["name"] = "processed_input"
            input_spec.append(spec)
        model_spec["input"] = input_spec
        return model_spec

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_input = inputs.get("processed_input", inputs.get("input", None))
        if model_input is None:
            raise KenningError("Cannot find input for the model")

        preds = self.runtime.infer(
            [model_input], self.model, postprocess=False
        )
        posty = self.model.postprocess_outputs(preds)

        io_spec = self.get_model_io_specification()

        result = {}
        for out_spec, out_value in zip(io_spec["output"], preds):
            result[out_spec["name"]] = out_value

        for out_spec, out_value in zip(io_spec["processed_output"], posty):
            result[out_spec["name"]] = out_value

        return result

    @staticmethod
    def _create_model_and_runtime(
        model_wrapper_parameters: Dict[str, Any],
        runtime_parameters: Dict[str, Any],
        dataset_parameters: Optional[Dict[str, Any]],
    ) -> Tuple[ModelWrapper, Runtime]:
        """
        Creates a ModelWrapper instance and a Runtime instance, optionally with
        a Dataset instance injected.

        Parameters
        ----------
        model_wrapper_parameters : Dict[str, Any]
            JSON describing the ModelWrapper object, following its argument
            structure
        runtime_parameters : Dict[str, Any]
            JSON describing the Runtime object, following its argument
            structure.
        dataset_parameters: Optional[Dict[str, Any]]
            JSON describing the Dataset object, following its argument
            structure.

        Returns
        -------
        Tuple[ModelWrapper, Runtime]
            The ModelWrapper object and the Runtime object, respectively.
        """
        config = {
            "model_wrapper": model_wrapper_parameters,
            "runtime": runtime_parameters,
        }
        if dataset_parameters:
            config["dataset"] = dataset_parameters
        config = yaml_or_json_to_config_dict(config)
        objs = objs_from_full_dict_config(config)
        return objs[ConfigKey.model_wrapper], objs[ConfigKey.runtime]

    def cleanup(self):
        self.runtime.inference_session_end()

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
