# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides an API for AutoML flow.
"""

from abc import ABC, abstractmethod
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.core.platform import Platform
from kenning.utils.args_manager import (
    ArgumentsHandler,
    get_type,
    supported_keywords,
    traverse_parents_with_args,
)


class AutoMLInvalidSchemaError(Exception):
    """
    Raised when `arguments_structure` contains not enough information
    or when data are invalid.
    """

    ...


class AutoMLInvalidArgumentsError(Exception):
    """
    Raised when provided arguments (in `use_model`) do not match with
    model wrapper `arguments_structure`.
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
        for name, config in schema.items():
            if config.get("type", None) is None or "default" not in config:
                raise AutoMLInvalidSchemaError(
                    f"{cls.__name__}:{config} misses `type` or `default`"
                )

            _type, sub_type = get_type(config["type"])
            if _type is list:
                if not sub_type:
                    raise AutoMLInvalidSchemaError(
                        f"{cls.__name__}:{name} misses list element type"
                    )
                if len(sub_type) > 1:
                    raise AutoMLInvalidSchemaError(
                        f"{cls.__name__}:{name} list contains"
                        " more than one sub-type"
                    )
                if any(isinstance(t, tuple) for t in sub_type):
                    raise AutoMLInvalidSchemaError(
                        f"{cls.__name__}:{name} union types ({_type}) "
                        "are not supported for AutoML params"
                    )
                sub_type = sub_type[0]

                if (
                    sub_type in (int, float)
                    and config.get("list_range", None) is None
                    and all(
                        config.get(arg, None) is None
                        for arg in ("enum", "item_range")
                    )
                ):
                    raise AutoMLInvalidSchemaError(
                        f"{cls.__name__}:{name} has to define "
                        "`list_range` and `item_range` or `enum`"
                    )
                if sub_type is str and any(
                    config.get(arg, None) is None
                    for arg in ("list_range", "enum")
                ):
                    raise AutoMLInvalidSchemaError(
                        f"{cls.__name__}:{name} has to define "
                        "`list_range` and `enum`"
                    )
            if config.get("enum", None) is None:
                if (
                    _type in (int, float)
                    and config.get("item_range", None) is None
                ):
                    raise AutoMLInvalidSchemaError(
                        f"{cls.__name__}:{name} has to define "
                        "`item_range` or `enum`"
                    )
                if _type is str:
                    raise AutoMLInvalidSchemaError(
                        f"{cls.__name__}:{name} has to define `enum`"
                    )
                if config.get("nullable", False):
                    raise AutoMLInvalidSchemaError(
                        f"{cls.__name__}:{name} can only be nullable for enum values"  # noqa: E501
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

    @classmethod
    def update_automl_range(cls, name: str, conf: Dict[str, Tuple]):
        """
        Updates the ranges of AutoML parameters.

        Parameters
        ----------
        name : str
            The name of parameter, should match with the `arguments_structure`.
        conf : Dict[str, Tuple]
            The dictionary with new parameters configuration.

        Raises
        ------
        AutoMLInvalidArgumentsError
            If parameter or its configuration does not exist.
        """
        arg_structure = {}
        for cls_ in reversed(list(traverse_parents_with_args(cls))):
            arg_structure |= cls_.arguments_structure

        if name not in arg_structure:
            raise AutoMLInvalidArgumentsError(
                f"Class `{cls.__name__}` does not have `{name}` parameter"
            )
        for conf_key, conf_value in conf.items():
            if (
                conf_key not in arg_structure[name]
                and conf_key not in supported_keywords
            ):
                raise AutoMLInvalidArgumentsError(
                    f"Parameter `{name}` of class `{cls.__name__}` "
                    f"does not have `{conf}` option"
                )
            arg_structure[name][conf_key] = conf_value
        cls.update_automl_defaults(arg_structure, name)

    @staticmethod
    def update_automl_defaults(arg_structure: Dict[str, Dict], name: str):
        """
        Updates the default value of the AutoML parameter
        to make sure they fit into ranges.

        Parameters
        ----------
        arg_structure : Dict[str, Dict]
            The combined arguments structure of current class and its parents.
        name : str
            The name of parameter, should match with the `arguments_structure`.
        """
        arg = arg_structure[name]
        _type, _ = get_type(arg["type"])
        default = arg["default"]
        if "enum" in arg and _type is not list:
            enum = arg["enum"]
            if default not in enum:
                arg["default"] = enum[0]
        elif _type is int or _type is float:
            range = arg["item_range"]
            if not (range[0] <= default <= range[1]):
                arg["default"] = range[0]
        elif _type is list:
            range = arg.get("enum", arg["item_range"])
            len_range = arg["list_range"]
            if len_range[0] >= len(default):
                default += [range[0]] * (len_range[0] - len(default))
            elif len_range[1] <= len(default):
                default = default[: (len_range[1])]
            for i, v in enumerate(default):
                if (
                    "item_range" in arg and not (range[0] <= v <= range[1])
                ) or ("enum" in arg and v not in range):
                    default[i] = range[0]
            arg["default"] = default


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
            "type": list[object | str],
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
        platform: Platform,
        output_directory: Path,
        optimizers: List[Optimizer] = [],
        use_models: List[Union[str, Dict[str, Tuple]]] = [],
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
        platform : Platform
            Platform on which generated models will be evaluated.
        output_directory : Path
            The path to the directory where found models
            and their measurements will be stored.
        optimizers : List[Optimizer]
            List of Optimizer objects that optimize the model.
        use_models : List[Union[str, Dict[str, Tuple]]]
            List of either:
                * class paths or names of models wrapper to use,
            classes have to implement AutoMLModel, or
                * dictionaries with class path/name as a key
            and overrides for AutoML parameters ranges.
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
        self.platform = platform
        self.output_directory = output_directory
        self.optimizers = optimizers
        self.time_limit = time_limit
        self.optimize_metric = optimize_metric
        self.n_best_models = n_best_models
        self.seed = seed
        self.jobs = 1
        if not use_models:
            use_models = self.supported_models
        self.use_models: List[Type[AutoMLModel]] = [
            load_class_by_type(
                module_path
                if isinstance(module_path, str)
                else list(module_path.keys())[0],
                MODEL_WRAPPERS,
            )
            for module_path in use_models
        ]
        assert all(
            [
                issubclass(cls, AutoMLModel) and issubclass(cls, ModelWrapper)
                for cls in self.use_models
            ]
        ), "All provided classes in `use_models` have to inherit from AutoMLModel"  # noqa: E501

        # Override ranges of AutoML parameter
        for i, (_class, model_with_conf) in enumerate(
            zip(self.use_models, use_models)
        ):
            if not isinstance(model_with_conf, dict):
                continue
            self.use_models[i] = _class = deepcopy(_class)
            for key, conf in list(model_with_conf.values())[0].items():
                _class.update_automl_range(key, conf)

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
    def get_best_configs(self) -> Iterable[Dict]:
        """
        Extracts the best models and returns Kenning configuration for them.

        Yields
        ------
        Dict
            Configuration for found models (from the best one).
        """
        ...

    @classmethod
    def from_argparse(
        cls,
        dataset: Optional[Dataset],
        platform: Optional[Platform],
        args: Namespace,
    ) -> "AutoML":
        """
        Constructor wrapper that takes the parameters from argparse args.

        Parameters
        ----------
        dataset : Optional[Dataset]
            The dataset object that is optionally used for optimization.
        platform : Optional[Platform]
            The platform on which generated models will be evaluated.
        args : Namespace
            Arguments from ArgumentParser object.

        Returns
        -------
        AutoML
            Object of class AutoML.
        """
        return super().from_argparse(args, dataset=dataset, platform=platform)

    @classmethod
    def from_json(
        cls,
        json_dict: Dict,
        dataset: Optional[Dataset] = None,
        platform: Optional[Platform] = None,
        optimizers: Optional[List[Optimizer]] = None,
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
        platform : Optional[Platform]
            The platform on which generated models will be evaluated.
        optimizers : Optional[List[Optimizer]]
            The optional list with optimizers.

        Returns
        -------
        AutoML
            Object of class AutoML.
        """
        return super().from_json(
            json_dict,
            dataset=dataset,
            platform=platform,
            optimizers=optimizers,
        )
