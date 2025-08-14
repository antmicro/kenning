# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides an API for AutoML flow.
"""

from abc import ABC, abstractmethod
from argparse import Namespace
from copy import copy, deepcopy
from logging import Logger
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from kenning.core.dataset import Dataset
from kenning.core.exceptions import (
    InvalidArgumentsError,
    InvalidSchemaError,
    ModelSizeError,
)
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import OptimizedModelSizeError, Optimizer
from kenning.core.platform import Platform
from kenning.core.runtime import Runtime
from kenning.runtimes.utils import get_default_runtime
from kenning.utils.args_manager import (
    ArgumentsHandler,
    get_type,
    supported_keywords,
    traverse_parents_with_args,
)
from kenning.utils.logger import KLogger


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
        InvalidSchemaError
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
                raise InvalidSchemaError(
                    f"{cls.__name__}:{config} misses `type` or `default`"
                )

            _type, sub_type = get_type(config["type"])
            if _type is list:
                if not sub_type:
                    raise InvalidSchemaError(
                        f"{cls.__name__}:{name} misses list element type"
                    )
                if len(sub_type) > 1:
                    raise InvalidSchemaError(
                        f"{cls.__name__}:{name} list contains"
                        " more than one sub-type"
                    )
                if any(isinstance(t, tuple) for t in sub_type):
                    raise InvalidSchemaError(
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
                    raise InvalidSchemaError(
                        f"{cls.__name__}:{name} has to define "
                        "`list_range` and `item_range` or `enum`"
                    )
                if sub_type is str and any(
                    config.get(arg, None) is None
                    for arg in ("list_range", "enum")
                ):
                    raise InvalidSchemaError(
                        f"{cls.__name__}:{name} has to define "
                        "`list_range` and `enum`"
                    )
            if config.get("enum", None) is None:
                if (
                    _type in (int, float)
                    and config.get("item_range", None) is None
                ):
                    raise InvalidSchemaError(
                        f"{cls.__name__}:{name} has to define "
                        "`item_range` or `enum`"
                    )
                if _type is str:
                    raise InvalidSchemaError(
                        f"{cls.__name__}:{name} has to define `enum`"
                    )
                if config.get("nullable", False):
                    raise InvalidSchemaError(
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
        InvalidArgumentsError
            If parameter or its configuration does not exist.
        """
        arg_structure = {}
        for cls_ in reversed(list(traverse_parents_with_args(cls))):
            arg_structure |= cls_.arguments_structure

        if name not in arg_structure:
            raise InvalidArgumentsError(
                f"Class `{cls.__name__}` does not have `{name}` parameter"
            )
        for conf_key, conf_value in conf.items():
            if (
                conf_key not in arg_structure[name]
                and conf_key not in supported_keywords
            ):
                raise InvalidArgumentsError(
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
        "application_size": {
            "description": "The size of an application running on the platform (in KB). If platform has restricted amount of RAM, AutoML flow will train only models that fit into the platform",  # noqa: E501
            "type": float,
            "nullable": True,
            "default": None,
        },
        "skip_model_size_check": {
            "description": "Whether the optimized model size check should be skipped",  # noqa: E501
            "type": bool,
            "default": False,
        },
        "callback_max_samples": {
            "description": "The maximum number of samples from dataset, which can be used in pre_training_callback method",  # noqa: E501
            "type": int,
            "default": 30,
        },
        "seed": {
            "description": "The seed used for AutoML",
            "type": int,
            "default": 1234,
        },
    }

    # File name for the JSON with AutoML statistics
    STATS_FILE_NAME = "automl_statistics.json"

    def __init__(
        self,
        dataset: Dataset,
        platform: Platform,
        output_directory: Path,
        optimizers: List[Optimizer] = [],
        runtime: Optional[Runtime] = None,
        use_models: List[Union[str, Dict[str, Tuple]]] = [],
        time_limit: float = 5.0,
        optimize_metric: str = "accuracy",
        n_best_models: int = 5,
        application_size: Optional[float] = None,
        skip_model_size_check: bool = False,
        callback_max_samples: int = 30,
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
        runtime : Optional[Runtime]
            The runtime used for models evaluation.
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
        application_size : Optional[float]
            The size of an application (in KB) run on the platform.
            If platform has restricted amount of RAM, AutoML will train
            only models that fit into the platform.
        skip_model_size_check : bool
            Whether the optimized model size check should be skipped.
        callback_max_samples : int
            The maximum number of samples from dataset,
            which can be used in pre_training_callback method.
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
        self.runtime = runtime
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

        # Prepare reduced dataset for pre-training callback
        self.application_size = application_size
        if self.application_size is None:
            self.application_size = 0
        self.skip_model_size_check = skip_model_size_check
        self.reduced_dataset = self.dataset
        if not self.skip_model_size_check:
            if callback_max_samples:
                self.reduced_dataset = copy(self.dataset)
                self.reduced_dataset.dataset_percentage = min(
                    callback_max_samples, len(self.dataset)
                ) / len(self.dataset)
                if self.reduced_dataset.dataset_percentage < 1.0:
                    self.reduced_dataset._reduce_dataset()

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

    @abstractmethod
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Returns statistic of the AutoML flow,
        like number of successful or crashed runs.

        The created dictionary has to contain
        "general_info" - mapping of statisctics descriptions and values.

        Optional fields:
        * "trained_model_metrics" - mapping of models to dictionaries
          with datasets and metrics of trained models,
        * "training_data" - mapping of models to losses from different
          parts of training, containing dictionaries with timestamps
          and loss values (averaged from batch or whole epoch). Possible
          parts of training: "training", "training_epoch", "validation",
          "validation_epoch", "test" and "test_epoch",
        * "training_start_time" - mapping of models to a list of times,
          marking the beginning of trainings,
        * "model_params" - mapping of models to dictionaries with parameters
          descriptions and values.

        Returns
        -------
        Dict[str, Union[int, float]]
            Dictionary with AutoML statistics,
            keys should describe given statistic.
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

    def _pre_training_callback(
        self,
        model_wrapper_cls: Type[ModelWrapper],
        model: Callable,
        logger: Logger = KLogger,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Prepares PipelineRunner and triggers the defined optimizations.
        It also extracts the optimized model size
        and available space on a platform.

        Parameters
        ----------
        model_wrapper_cls : Type[ModelWrapper]
            The Kenning ModelWrapper class representing a model architecture.
        model : Callable
            The actual model object, not a ModelWrapper.
        logger : Logger
            The logger used by a AutoML framework.

        Returns
        -------
        Optional[float]
            The optimized model size.
        Optional[float]
            The available space on the platform.

        Raises
        ------
        ModelSizeError
            If model size is invalid from other reasons
            than not fitting into the platform memory.
        """
        if self.skip_model_size_check:
            return None, None

        from kenning.optimizers.ai8x import Ai8xIzerError
        from kenning.utils.pipeline_runner import PipelineRunner

        available_size = None
        model_size = None
        with NamedTemporaryFile("w") as fd:
            # Save model to file
            model_wrapper: ModelWrapper = model_wrapper_cls(
                Path(fd.name), self.reduced_dataset
            )
            model_wrapper.model = model
            model_wrapper.model_prepared = True
            model_wrapper.save_model(model_wrapper.model_path)

            # Prepare PipelineRunner and get available_size
            runtime = self.runtime
            runner = PipelineRunner(
                dataset=self.reduced_dataset,
                optimizers=self.optimizers,
                runtime=runtime,
                platform=self.platform,
                model_wrapper=model_wrapper,
            )
            if runtime is None:
                model_framework = runner._guess_model_framework(False)
                runtime = get_default_runtime(
                    model_framework, model_wrapper.model_path
                )
                runner.runtime = runtime
            if runtime:
                available_size = runtime.get_available_ram(self.platform)

            # Run Kenning optimize flow for the model
            if self.optimizers:
                try:
                    for opt in self.optimizers:
                        opt.model_wrapper = model_wrapper
                        opt.dataset = self.reduced_dataset
                    runner._handle_optimizations()
                except Ai8xIzerError as ex:
                    raise ModelSizeError(ex.model_size) from ex
                finally:
                    for opt in self.optimizers:
                        opt.model_wrapper = None
                        opt.dataset = self.dataset
                try:
                    model_size = runner.optimizers[
                        -1
                    ].get_optimized_model_size()
                except OptimizedModelSizeError as ex:
                    logger.warning(
                        f"Cannot retrieve optimized model size: {ex}"
                    )
                # Cleanup optimized models
                for opt in self.optimizers:
                    opt.compiled_model_path.unlink()
            else:
                model_size = model_wrapper.get_model_size()
        return model_size, available_size
