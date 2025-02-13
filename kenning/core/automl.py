# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides an API for AutoML flow.
"""

from abc import ABC, abstractmethod
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.utils.args_manager import ArgumentsHandler


class AutoMLInvalidSchemaError(Exception):
    """
    Raised when `arguments_structure` contains not enough information
    or when data are invalid.
    """

    ...


class AutoMLModel(ArgumentsHandler, ABC):
    """
    Base class representing AutoML-compatible model.

    Should be used together with kenning.core.model.ModelWrapper class.
    """

    @classmethod
    def form_automl_schema(cls) -> Dict:
        """
        Gathers AutoML schema based on `arguments_structure` of class
        and all parent.

        Returns
        -------
        Dict
            AutoML schema for the class.

        Raises
        ------
        AutoMLInvalidSchemaError
            If schema does not contain required fields.
        """
        classes = [cls]
        schema = {}

        while len(classes):
            curr_cls = classes.pop(0)
            classes.extend(curr_cls.__bases__)
            if not hasattr(curr_cls, "arguments_structure"):
                continue
            automl_params = {
                k: v
                for k, v in curr_cls.arguments_structure.items()
                if v.get("AutoML", False)
            }
            schema = automl_params | schema
        # Check whether merged schema contains necessary params
        for config in schema.values():
            if any(
                [config.get(arg, None) is None for arg in ("type", "default")]
            ):
                raise AutoMLInvalidSchemaError(
                    f"{cls.__name__}:{config} misses `type` or `default`"
                )
            _type = config["type"]
            if _type is list and any(
                config.get(arg, None) is None
                for arg in ("list_range", "items", "item_range")
            ):
                raise AutoMLInvalidSchemaError(
                    f"{cls.__name__}:{config} has to define "
                    "`list_range`, `items` and `item_range`"
                )
            if (
                _type in (int, float)
                and config.get("item_range", None) is None
            ):
                raise AutoMLInvalidSchemaError(
                    f"{cls.__name__}:{config} has to define `item_range`"
                )
            if _type is str and config.get("enum", None) is None:
                raise AutoMLInvalidSchemaError(
                    f"{cls.__name__}:{config} has to define `enum`"
                )

        return schema

    @abstractmethod
    def extract_model(self, network: Any) -> Any:
        """
        Extracts model from AutoML object.

        Parameters
        ----------
        network : Any
            AutoML object with optimized model.

        Returns
        -------
        Any
            Extracted model
        """
        ...


class AutoML(ArgumentsHandler, ABC):
    """
    Base class describing AutoML flow:
    * preparing AutoML framework,
    * searching the best models,
    * generating Kenning configs based on found solutions.
    """

    # Default list of supported models (have to inherit from AutoMLModel)
    supported_models: List[str] = []

    arguments_structure = {
        "output_directory": {
            "description": "The path to the directory where found models and their measurements will be stored",  # noqa: E501
            "type": Path,
            "required": True,
        },
        "use_models": {
            "description": "The list of class paths or names of models wrapper to use, classes have to implement AutoMLModel",  # noqa: E501
            "type": list,
            "items": str,
            "default": [],
        },
        "time_limit": {
            "description": "The time limit in minutes",
            "type": float,
            "default": 5.0,
        },
        "optimize_metric": {
            "description": "The metric to optimize",
            "type": str,
            "default": "accuracy",
        },
        "n_best_models": {
            "description": "The upper limit of number of models to return",
            "type": int,
            "default": 5,
        },
        "seed": {
            "description": "The seed used for AutoML",
            "type": int,
            "default": 1234,
        },
    }

    def __init__(
        self,
        dataset: Dataset,
        output_directory: Path,
        use_models: List[str] = [],
        time_limit: float = 5.0,
        optimize_metric: str = "accuracy",
        n_best_models: int = 5,
        seed: int = 1234,
    ):
        """
        Prepares the AutoML object.

        Parameters
        ----------
        dataset : Dataset
            Dataset for which models will be optimized.
        output_directory : Path
            The path to the directory where found models
            and their measurements will be stored.
        use_models : List[str]
            The list of class paths or names of models wrapper to use,
            classes have to implement AutoMLModel.
        time_limit : float
            The time limit in minutes.
        optimize_metric : str
            The metric to optimize.
        n_best_models : int
            The upper limit of number of models to return.
        seed : int
            The seed used for AutoML.
        """
        from kenning.utils.class_loader import (
            MODEL_WRAPPERS,
            load_class_by_type,
        )

        self.dataset = dataset
        self.output_directory = output_directory
        self.time_limit = time_limit
        self.optimize_metric = optimize_metric
        self.n_best_models = n_best_models
        self.seed = seed
        self.jobs = 1
        if not use_models:
            use_models = self.supported_models
        self.use_models: List[Type[AutoMLModel]] = [
            load_class_by_type(module_path, MODEL_WRAPPERS)
            for module_path in use_models
        ]
        assert all(
            [
                issubclass(cls, AutoMLModel) and issubclass(cls, ModelWrapper)
                for cls in self.use_models
            ]
        ), "All provided classes in `use_models` have to inherit from AutoMLModel"  # noqa: E501
        self.output_directory.mkdir(exist_ok=True, parents=True)

    @abstractmethod
    def prepare_framework(self):
        """
        Prepares AutoML framework.
        """
        ...

    @abstractmethod
    def search(self):
        """
        Runs AutoML search.
        """
        ...

    @abstractmethod
    def get_best_configs(self) -> List[Dict]:
        """
        Extracts the best models and returns Kenning configuration for them.

        Returns
        -------
        List[Dict]
            Configurations for found models (from the best one).
        """
        ...

    @classmethod
    def from_argparse(
        cls,
        dataset: Optional[Dataset],
        args: Namespace,
    ) -> "AutoML":
        """
        Constructor wrapper that takes the parameters from argparse args.

        Parameters
        ----------
        dataset : Optional[Dataset]
            The dataset object that is optionally used for optimization.
        args : Namespace
            Arguments from ArgumentParser object.

        Returns
        -------
        AutoML
            Object of class AutoML.
        """
        return super().from_argparse(args, dataset=dataset)

    @classmethod
    def from_json(
        cls,
        json_dict: Dict,
        dataset: Optional[Dataset] = None,
    ) -> "AutoML":
        """
        Constructor wrapper that takes the parameters from json dict.

        This function checks if the given dictionary is valid according
        to the `arguments_structure` defined.
        If it is then it invokes the constructor.

        Parameters
        ----------
        json_dict : Dict
            Arguments for the constructor.
        dataset : Optional[Dataset]
            The dataset object that is optionally used for optimization.

        Returns
        -------
        AutoML
            Object of class AutoML.
        """
        return super().from_json(json_dict, dataset=dataset)
