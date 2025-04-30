# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides an API for AutoML flow with AutoPyTorch framework.
"""

from collections import defaultdict
from copy import copy
from multiprocessing import cpu_count
from pathlib import Path
from shutil import rmtree
from tempfile import NamedTemporaryFile
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import psutil
from sklearn.pipeline import Pipeline

from kenning.core.automl import AutoML, AutoMLModel
from kenning.core.dataset import Dataset
from kenning.core.optimizer import Optimizer
from kenning.core.platform import Platform
from kenning.utils.args_manager import get_type, traverse_parents_with_args
from kenning.utils.class_loader import load_class
from kenning.utils.logger import KLogger
from kenning.utils.pipeline_runner import PipelineRunner

TOTAL_RAM = psutil.virtual_memory().total // 1024
BUDGET_TYPES = ["epochs", "runtime"]
# Type variables representing real classes
PyTorchModel = TypeVar("torch.nn.Module")
ConfigurationSpace = TypeVar("ConfigSpace.ConfigurationSpace")
NetworkBackboneComponent = TypeVar(
    "autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone.NetworkBackboneComponent"
)


class MissingConfigForAutoPyTorchModel(Exception):
    """
    Raised when required configuration to initialize
    AutoPyTorch model was not provided.
    """

    ...


class ModelExtractionError(Exception):
    """
    Raised when Kenning model was not properly extracted from AutoPyTorch.
    """

    ...


class ModelClassNotValid(Exception):
    """
    Raised when provided model class cannot be imported.
    """

    ...


class AutoPyTorchModel(AutoMLModel):
    """
    Base class representing AutoPyTorch-compatible model.

    Should be used together with kenning.core.model.ModelWrapper class.

    It automatically creates AutoPyTorch component representing the model
    and its hyperparameters based on the `arguments_structure`.
    Also, it extracts model and prepares corresponding Kenning configuration.
    """

    # Defines the name of the PyTorch class with model,
    # required for model extraction
    model_class: str = None

    @classmethod
    def get_component_name(cls) -> str:
        """
        Returns name of created component class.

        Returns
        -------
        str
            Name of created component class.
        """
        return f"{cls.__name__}_Component"

    @classmethod
    def get_hyperparameter_search_space(
        cls,
        dataset_properties: Optional[Dict] = None,
    ) -> ConfigurationSpace:
        """
        Generates Configuration Space based on `arguments_structure`.

        Parameters
        ----------
        dataset_properties : Optional[Dict]
            Properties of used dataset, provided by AutoPyTorch.

        Returns
        -------
        ConfigurationSpace
            Configuration Space with model hyperparameter.
        """
        import ConfigSpace as CS

        from kenning.automl.auto_pytorch_components.utils import (
            _add_single_hyperparameter,
        )

        args = cls.form_automl_schema()
        cs = CS.ConfigurationSpace()
        for name, config in args.items():
            c_type, item_type = get_type(config["type"])
            c_default = config["default"]
            if c_type in (int, float, str, bool):
                _add_single_hyperparameter(cs, name, config, c_type, c_default)
            else:  # c_type is list
                list_range = config["list_range"]
                list_min_len, list_max_len = list_range
                list_len_param = _add_single_hyperparameter(
                    cs, name, config, c_type, len(c_default)
                )

                for i in range(list_max_len):
                    i_value_param = _add_single_hyperparameter(
                        cs,
                        f"{name}_{i}",
                        config,
                        item_type[0],
                        c_default[min(i, len(c_default) - 1)],
                    )
                    # Add condition for element of the list
                    if i + 1 > list_min_len:
                        cs.add_condition(
                            CS.GreaterThanCondition(
                                i_value_param, list_len_param, i
                            )
                        )

        return cs

    @staticmethod
    def _create_model_structure(
        input_shape: Iterable[int],
        dataset: Optional[Dataset] = None,
    ) -> PyTorchModel:
        """
        Recreates the model structure.

        Parameters
        ----------
        input_shape : Iterable[int]
            The shape of input data.
        dataset : Optional[Dataset]
            Dataset used for the model.

        Returns
        -------
        PyTorchModel
            Created PyTorch model.

        Raises
        ------
        NotImplementedError
            If method has not been implemented in child class.
        """
        raise NotImplementedError

    @classmethod
    def define_forbidden_clauses(
        cls,
        cs: ConfigurationSpace,
        **kwargs: Dict,
    ) -> ConfigurationSpace:
        """
        Defines forbidden clauses for not compatible pairs of hyperparametrs.

        By default, it disables all preprocessing features apart from scaling.

        Parameters
        ----------
        cs : ConfigurationSpace
            Configuration Space with hyperparameters.
        **kwargs : Dict
            Additional parameters.

        Returns
        -------
        ConfigurationSpace
            Updated Configuration Space with forbidden clauses.
        """
        import ConfigSpace as CS

        from kenning.automl.auto_pytorch_components.utils import (
            _create_forbidden_choices,
        )

        model_component = cls.get_component_name()
        network_back = cs.get_hyperparameter("network_backbone:__choice__")
        network_back = CS.ForbiddenEqualsClause(
            network_back,
            model_component,
        )

        clauses = [
            _create_forbidden_choices(cs, name, (choice,), True)
            for name, choice in (
                ("imputer:numerical_strategy", "constant_zero"),
                ("network_head:__choice__", "PassthroughHead"),
                ("network_embedding:__choice__", "NoEmbedding"),
                ("feature_preprocessor:__choice__", "NoFeaturePreprocessor"),
                ("encoder:__choice__", "NoEncoder"),
                ("coalescer:__choice__", "NoCoalescer"),
                ("scaler:__choice__", "StandardScaler"),
            )
        ]

        cs.add_forbidden_clauses(
            [
                CS.ForbiddenAndConjunction(
                    network_back,
                    clause,
                )
                for clause in clauses
            ]
        )
        return cs

    @classmethod
    def model_params_from_context(
        cls, dataset: Dataset, platform: Optional[Platform] = None
    ) -> Dict[str, Any]:
        """
        Extracts additional model parameters based on context
        like dataset and platform.

        Parameters
        ----------
        dataset : Dataset
            The dataset used for processing.
        platform : Optional[Platform]
            The platform used for evaluation.

        Returns
        -------
        Dict[str, Any]
            Generated dictionary with model parameters.
        """
        return {}

    @classmethod
    def build_backbone(
        cls,
        self: NetworkBackboneComponent,
        input_shape: Tuple[int, ...],
        dataset: Optional[Dataset] = None,
        platform: Optional[Platform] = None,
        processed_input: Optional[Tuple[int, ...]] = None,
    ) -> PyTorchModel:
        """
        AutoPyTorch component method preparing model object.

        Parameters
        ----------
        self : NetworkBackboneComponent
            Created AutoPyTorch component.
        input_shape : Tuple[int, ...]
            The shape of input data.
        dataset : Optional[Dataset]
            The dataset used for the model.
        platform : Optional[Platform]
            The platform used for the model.
        processed_input: Optional[Tuple[int, ...]]
            The shape of input that should be provided to the model.

        Returns
        -------
        PyTorchModel
            Created PyTorch model.

        Raises
        ------
        MissingConfigForAutoPyTorchModel
            If config requaries for model is missing.
        """
        assert (
            input_shape is not None
        ), "AutoPyTorch has not provided input shape"

        from torch.nn import Sequential, Unflatten

        model_class_obj = load_class(cls.model_class)
        if model_class_obj is None:
            raise ModelClassNotValid(f"{cls.model_class} cannot be imported")
        schema = cls.form_automl_schema()
        args = {}
        for name, config in schema.items():
            if name not in self.config:
                raise MissingConfigForAutoPyTorchModel(
                    f"Missing {name} config"
                )
            args[name] = self.config[name]
            _type, _ = get_type(config.get("type", None))
            if _type is not list:
                continue
            args[name] = [
                self.config.get(f"{name}_{i}", None)
                for i in range(self.config[name])
            ]
            if not all(args[name]):
                raise MissingConfigForAutoPyTorchModel(
                    f"Missing values in {name} config"
                )
        model = model_class_obj(
            **args,
            input_shape=input_shape,
            **cls.model_params_from_context(dataset, platform),
        )
        if processed_input and input_shape != processed_input[1:]:
            KLogger.info(
                "Adding Unflatten layer to match model inupt shape - "
                f"{input_shape} -> {processed_input[1:]}"
            )
            return Sequential(
                Unflatten(-1, processed_input[1:]),
                model,
            )
        return model

    @classmethod
    def get_properties(
        cls,
        dataset_properties: Optional[Dict] = None,
    ) -> Dict[str, Union[str, bool]]:
        """
        AutoPyTorch component method returning its properties,
        like name and supported type of data.

        Parameters
        ----------
        dataset_properties : Optional[Dict]
            Properties of used dataset, provided by AutoPyTorch.

        Returns
        -------
        Dict[str, Union[str, bool]]
            Component properties.
        """
        name = cls.get_component_name()
        return {
            "shortname": name,
            "name": name,
            "handles_tabular": True,
            "handles_image": False,
            "handles_time_series": False,
        }

    @classmethod
    def get_io_specification_from_dataset(
        cls, dataset: Dataset
    ) -> Dict[str, List[Dict]]:
        """
        Creates IO specification based on the Dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset used for the model.

        Returns
        -------
        Dict[str, List[Dict]]
            Prepared IO specification.

        Raises
        ------
        NotImplementedError
            If method has not been implemented in child class.
        """
        raise NotImplementedError

    @classmethod
    def register_components(
        cls,
        dataset: Dataset,
        platform: Platform,
    ) -> List[Type[NetworkBackboneComponent]]:
        """
        Dynamically creates new AutoPyTorch component classes,
        based on NetworkBackboneComponent, and register them.

        Parameters
        ----------
        dataset : Dataset
            The dataset used for the model.
        platform : Platform
            The platform used for the model.

        Returns
        -------
        List[Type[NetworkBackboneComponent]]
            List of created and registered components.
        """
        from autoPyTorch.pipeline.components.setup.network_backbone import (
            add_backbone,
        )
        from autoPyTorch.pipeline.components.setup.network_backbone.base_network_backbone import (  # noqa: E501
            NetworkBackboneComponent,
        )

        from kenning.automl.auto_pytorch_components.network_head_passthrough import (  # noqa: E501
            register_passthrough,
        )

        io_spec = cls.get_io_specification_from_dataset(cls, dataset)
        processed_input = io_spec.get("processed_input", io_spec["input"])[0][
            "shape"
        ]

        component = type(
            cls.get_component_name(),
            (NetworkBackboneComponent,),
            {
                "get_hyperparameter_search_space": staticmethod(
                    cls.get_hyperparameter_search_space
                ),
                "define_forbidden_clauses": staticmethod(
                    cls.define_forbidden_clauses
                ),
                "build_backbone": lambda self,
                *args,
                **kwargs: cls.build_backbone(
                    self,
                    *args,
                    **kwargs,
                    dataset=dataset,
                    platform=platform,
                    processed_input=processed_input,
                ),
                "get_properties": staticmethod(cls.get_properties),
            }
            | (
                {
                    "fit": lambda self, *args, **kwargs: cls.fit(
                        self, *args, **kwargs
                    ),
                }
                if hasattr(cls, "fit")
                else {}
            ),
        )
        add_backbone(component)
        # Register passthrough head to not change the model output
        register_passthrough()
        return [component]

    @classmethod
    def extract_model(cls, network: PyTorchModel) -> PyTorchModel:
        """
        Extracts model from AutoPyTorch component.

        Parameters
        ----------
        network : PyTorchModel
            The final network found by AutoPyTorch.

        Returns
        -------
        PyTorchModel
            Extracted Kenning compatible model.
        """
        model_class_obj = load_class(cls.model_class)
        for module in network.modules():
            if type(module).__name__ == model_class_obj.__name__:
                return module

    @classmethod
    def prepare_config(cls, configuration: Dict) -> Dict:
        """
        Prepares Kenning configuration based on configuration
        found by AutoPyTorch.

        Parameters
        ----------
        configuration : Dict
            Configuration found by AutoPyTorch.

        Returns
        -------
        Dict
            Kenning configuration.

        Raises
        ------
        MissingConfigForAutoPyTorchModel
            If AutoPyTorch configuration misses the required field.
        """
        args = cls.form_automl_schema()
        backbone_conf = {
            k.rsplit(":", 1)[1]: v
            for k, v in configuration.items()
            if k.startswith("network_backbone")
        }
        model_wrapper_params = {}
        for name, config in args.items():
            if name not in backbone_conf:
                raise MissingConfigForAutoPyTorchModel(
                    f"Missing {name} in AutoPyTorch config"
                )
            c_type, _ = get_type(config["type"])
            if backbone_conf[name] is None:
                continue
            if c_type is not list:
                model_wrapper_params[name] = backbone_conf[name]
            else:
                model_wrapper_params[name] = [
                    backbone_conf[f"{name}_{i}"]
                    for i in range(backbone_conf[name])
                ]
        # Append default non-AutoML params
        arg_structure = {}
        for cls_ in reversed(list(traverse_parents_with_args(cls))):
            arg_structure |= cls_.arguments_structure
        for name, config in arg_structure.items():
            if (
                name in model_wrapper_params
                or "default" not in config
                or config["default"] is None
            ):
                continue
            model_wrapper_params[name] = config["default"]
        return {
            "model_wrapper": {
                "type": f"{cls.__module__}.{cls.__name__}",
                "parameters": model_wrapper_params,
            },
        }


class AutoPyTorchML(AutoML):
    """
    Definition of AutoML flow with AutoPyTorch framework.
    """

    supported_models = [
        "kenning.modelwrappers.anomaly_detection.vae.PyTorchAnomalyDetectionVAE",
    ]

    arguments_structure = {
        "max_memory_usage": {
            "description": "Maximum system-wise RAM usage in KB, if exceeded AutoML subprocess will be terminated",  # noqa: E501
            "type": int,
            "default": TOTAL_RAM,
        },
        "max_evaluation_time": {
            "description": "The time limit (in minutes) for training and evaluation of one model, by default set to `time_limit`",  # noqa: E501
            "type": float,
            "default": None,
        },
        "budget_type": {
            "description": "Type of the budget for training model, either number of epoochs or time limit in seconds",  # noqa: E501
            "type": str,
            "enum": BUDGET_TYPES,
            "default": BUDGET_TYPES[0],
        },
        "min_budget": {
            "description": "The lower bound of the budget",
            "type": int,
            "default": 3,
        },
        "max_budget": {
            "description": "The upper bound of the budget",
            "type": int,
            "default": 10,
        },
        "application_size": {
            "description": "The size of an application running on the platform (in KB). If platform has restricted amount of RAM, AutoPyTorch will train only models that fit into the platform",  # noqa: E501
            "type": float,
            "default": None,
        },
        "callback_max_samples": {
            "description": "The maximum number of samples from dataset, which can be used in pre_training_callback method",  # noqa: E501
            "type": int,
            "default": 30,
        },
        "all_supported_metrics": {
            "description": "Calculate all supported metrics by AutoPyTorch during training",  # noqa: E501
            "type": bool,
            "default": True,
        },
        "use_cuda": {
            "description": "Whether to use GPU, if it is not available this option is ignored",  # noqa: E501
            "type": bool,
            "default": False,
        },
        "data_loader_workers": {
            "description": "The number of workers to use for data loaders",
            "type": int,
            "default": cpu_count() // 2,
        },
    }

    def __init__(
        self,
        dataset: Dataset,
        platform: Platform,
        output_directory: Path,
        optimizers: List[Optimizer] = [],
        use_models: List[str] = [],
        time_limit: float = 5.0,
        optimize_metric: str = "accuracy",
        n_best_models: int = 5,
        seed: int = 1234,
        max_memory_usage: int = TOTAL_RAM,
        max_evaluation_time: Optional[float] = None,
        budget_type: str = BUDGET_TYPES[0],
        min_budget: int = 3,
        max_budget: int = 10,
        application_size: Optional[float] = None,
        callback_max_samples: int = 30,
        all_supported_metrics: bool = True,
        use_cuda: bool = False,
        data_loader_workers: int = cpu_count() // 2,
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
        max_memory_usage : int
            Maximum system-wise RAM usage in KB,
            if exceeded AutoML subprocess will be terminated.
        max_evaluation_time : Optional[float]
            The time limit (in minutes) for training and
            evaluation of one model, by default set to `time_limit`.
        budget_type : str
            Type of the budget for training model,
            either number of epoochs ("epochs")
            or time limit ("runtime") in seconds.
        min_budget : int
            The lower bound of the budget.
        max_budget : int
            The upper bound of the budget.
        application_size : Optional[float]
            The size of an application (in KB) run on the platform.
            If platform has restricted amount of RAM, AutoPyTorch will train
            only models that fit into the platform.
        callback_max_samples : int
            The maximum number of samples from dataset,
            which can be used in pre_training_callback method.
        all_supported_metrics : bool
            Calculate all supported metrics by AutoPyTorch during training.
        use_cuda : bool
            Whether to use CUDA-compatible accelerator.
        data_loader_workers : int
            The number of workers to use for data loaders.
        """
        from kenning.modelwrappers.frameworks.pytorch import PyTorchWrapper

        super().__init__(
            dataset=dataset,
            platform=platform,
            output_directory=output_directory,
            optimizers=optimizers,
            use_models=use_models if use_models else self.supported_models,
            time_limit=time_limit,
            optimize_metric=optimize_metric,
            n_best_models=n_best_models,
            seed=seed,
        )
        assert all(
            [
                issubclass(model, AutoPyTorchModel)
                and issubclass(model, PyTorchWrapper)
                for model in self.use_models
            ]
        ), (
            "All provided classes in `use_models` have to "
            f"inherit from {AutoPyTorchModel.__name__} "
            f"and {PyTorchWrapper.__name__}"
        )
        assert (
            budget_type in BUDGET_TYPES
        ), f"Budget has to be one of {BUDGET_TYPES}"
        assert (
            callback_max_samples > 0
        ), "`callback_max_samples` has to be greater than 0"

        self.reduced_dataset = self.dataset

        self.use_models: List[AutoPyTorchModel]
        self.max_memory_usage = max_memory_usage
        self.max_evaluation_time = max_evaluation_time
        if max_evaluation_time is None:
            self.max_evaluation_time = self.time_limit
        self.budget_type = budget_type
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.application_size = application_size
        self.all_supported_metrics = all_supported_metrics
        self.use_cuda = use_cuda
        self.data_loader_workers = data_loader_workers

        self.platform_ram = getattr(self.platform, "ram_size_kb", None)
        self.max_model_size = None
        if self.application_size and self.platform_ram:
            assert (
                self.platform_ram > self.application_size
            ), f"Application ({self.application_size}) does not fit into the platform ({self.platform_ram})"  # noqa: E501
            self.max_model_size = self.platform_ram - self.application_size
            self.max_model_size *= 0.95
            if callback_max_samples:
                self.reduced_dataset = copy(self.dataset)
                self.reduced_dataset.dataset_percentage = min(
                    callback_max_samples, len(self.dataset)
                ) / len(self.dataset)
                if self.reduced_dataset.dataset_percentage < 1.0:
                    self.reduced_dataset._reduce_dataset()

        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self._components: List[Type[NetworkBackboneComponent]] = []
        self._prepared = False
        self._api = None

        self.initial_run_num: int = None
        self.model_paths: List[Path] = []
        self.best_configs: List[Path] = []

    def prepare_framework(self):
        # Split and flatten the dataset
        Xtr, Xte, Ytr, Yte = self.dataset.train_test_split_representations()
        self.X_train = np.asarray(
            self.dataset.prepare_input_samples(Xtr)
        ).reshape(len(Xtr), -1)
        self.X_test = np.asarray(
            self.dataset.prepare_input_samples(Xte)
        ).reshape(len(Xte), -1)
        self.y_train = np.asarray(
            self.dataset.prepare_output_samples(Ytr)
        ).reshape(len(Ytr), -1)
        self.y_test = np.asarray(
            self.dataset.prepare_output_samples(Yte)
        ).reshape(len(Yte), -1)

        # Register models components
        for model in self.use_models:
            self._components += model.register_components(
                self.dataset, self.platform
            )

        self._prepared = True

    def search(self):
        from autoPyTorch.api.tabular_classification import (
            TabularClassificationTask,
        )

        assert (
            self._prepared
        ), "`search` has to be called after `prepare_framework`"

        autoPyTorch_tmp_dir = self.output_directory / "_autoPyTorch_tmp"
        if autoPyTorch_tmp_dir.exists():
            KLogger.warn(
                f"Deleting autoPyTorch temporary folder: {autoPyTorch_tmp_dir}"
            )
            rmtree(autoPyTorch_tmp_dir)

        self._api = TabularClassificationTask(
            seed=self.seed,
            include_components={
                "network_backbone": [
                    component.__name__ for component in self._components
                ],
            },
            # search_space_updates=search_space_updates,
            n_jobs=self.jobs,
            n_threads=cpu_count(),
            # Do not ensemble models
            ensemble_nbest=1,
            ensemble_size=0,
            # Setup autoPyTorch directory, where intermediate results
            # and tensorboard data are stored
            temporary_directory=str(autoPyTorch_tmp_dir),
            delete_tmp_folder_after_terminate=False,
        )
        self._api.set_pipeline_options(
            device="cuda" if self.use_cuda else "cpu",
            torch_num_threads=cpu_count(),
            use_tensorboard_logger=True,
            early_stopping=True,
            pre_training_callback=self.pre_training_callback,
            data_loader_workers=self.data_loader_workers,
        )
        self.initial_run_num = self._api._backend.get_next_num_run()

        try:
            self._api.search(
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                y_test=self.y_test,
                optimize_metric=self.optimize_metric,
                total_walltime_limit=self.time_limit * 60,
                func_eval_time_limit_secs=self.max_evaluation_time * 60,
                memory_limit=self.max_memory_usage,
                budget_type=self.budget_type,
                min_budget=self.min_budget,
                max_budget=self.max_budget,
                # Disable non NN-based methods
                enable_traditional_pipeline=False,
                all_supported_metrics=self.all_supported_metrics,
            )
        except ValueError as e:
            if "No valid model" in str(e):
                KLogger.error(str(e))
            else:
                raise

    def extract_model(self, pipeline: Pipeline) -> PyTorchModel:
        """
        Extracts model from AutoPyTorch pipeline.

        Parameters
        ----------
        pipeline : Pipeline
            Fitted AutoPyTorch pipeline.

        Returns
        -------
        PyTorchModel
            Trained PyTorch model.
        """
        backbone = pipeline["network_backbone"].choice
        backbone_cls = type(backbone).__name__
        for mod_cls in self.use_models:
            if mod_cls.get_component_name() == backbone_cls:
                return (
                    mod_cls.extract_model(pipeline["network"].network),
                    mod_cls.prepare_config(pipeline.config),
                )

    def get_statistics(self) -> str:
        from autoPyTorch.pipeline.components.training.metrics.metrics import (
            CLASSIFICATION_METRICS,
        )
        from autoPyTorch.utils.results_manager import cost2metric

        stats = self._api.get_statistics()
        # Get metrics of trained models
        metrics_over_time = {}
        additional_info = {}
        for run_key, run_hist in self._api.run_history.items():
            if "num_run" not in run_hist.additional_info:
                KLogger.warning(
                    "`num_run` missing in run history "
                    f"with status {run_hist.status}"
                )
                continue
            num_run = int(run_hist.additional_info["num_run"])
            metrics_over_time[num_run] = {
                # Change losses to metrics
                k: {
                    m: cost2metric(v, CLASSIFICATION_METRICS[m])
                    for m, v in run_hist.additional_info[f"{k}_loss"].items()
                }
                for k in ("train", "opt", "test")
            }
            additional_info[num_run] = run_hist.additional_info

        # Get data from training with Tensorboard
        training_data = defaultdict(lambda: defaultdict(dict))
        training_start_time = defaultdict(list)
        model_params = defaultdict(lambda: defaultdict(dict))
        model_prefix = "Model/"
        EVENT_TO_NAME = {
            "Train/loss": "training",
            "Train/epoch/avg_loss": "training_epoch",
            "Val/loss": "validation",
            "Val/epoch/avg_loss": "validation_epoch",
            "Test/loss": "test",
            "Test/epoch/avg_loss": "test_epoch",
        }
        try:
            from tensorflow.core.util import event_pb2
            from tensorflow.data import TFRecordDataset
            from tensorflow.python.framework.errors_impl import DataLossError

            for events_file in Path(self._api._temporary_directory).glob(
                "events.out.tfevents.*"
            ):
                num_run = None
                KLogger.debug(events_file)
                try:
                    for event in TFRecordDataset(str(events_file)):
                        event = event_pb2.Event.FromString(event.numpy())
                        # Retrieve num_run as a model ID
                        if num_run is None:
                            for e_val in event.summary.value:
                                if e_val.tag == "num_run":
                                    num_run = int(e_val.simple_value)
                                    training_start_time[num_run].append(
                                        event.wall_time
                                    )
                                    break
                        # Gather metadata and data from training
                        else:
                            for e_val in event.summary.value:
                                if e_val.tag.startswith(model_prefix):
                                    v = e_val.simple_value
                                    if abs(v - int(v)) < 1e-8:
                                        v = int(v)
                                    model_params[num_run][
                                        e_val.tag[len(model_prefix) :]
                                    ] = v
                                else:
                                    training_data[num_run][
                                        EVENT_TO_NAME[e_val.tag]
                                    ][event.wall_time] = e_val.simple_value
                except DataLossError:
                    KLogger.warning(f"Possible data loss in {events_file}")
                    continue
        except ImportError:
            KLogger.warning(
                "Cannot import data from tensorboard,"
                " please install tensorflow"
            )
        return {
            "general_info": {
                "Optimized metric": self._api._metric.name,
                "The number of generated models": stats["runs"],
                "The number of trained and evaluated models": stats["success"],
                "The number of models that caused a crash": stats["crash"],
                "The number of models that failed due to the timeout": stats[
                    "timeout"
                ],
                "The number of models that failed due to the too large size": stats[  # noqa: E501
                    "memout"
                ],
            },
            "trained_model_metrics": metrics_over_time,
            "training_start_time": training_start_time,
            "model_params": model_params,
            "additional_info": additional_info,
        } | ({"training_data": training_data} if training_data else {})

    def get_best_configs(self) -> Iterable[Dict]:
        import torch
        from smac.tae import StatusType

        assert (
            self._api is not None
        ), "`get_best_config` has to be called after `search`"

        # Get n best results
        results = self._api.run_history.data
        results = [
            (r_key, r_value)
            for r_key, r_value in results.items()
            if r_value.status in (StatusType.SUCCESS, StatusType.DONOTADVANCE)
        ]
        results = sorted(
            results,
            key=lambda x: x[1].additional_info["opt_loss"][
                self.optimize_metric
            ],
        )

        self.best_configs = []
        # Load pipelines from results, extract models and save them
        for r_key, r_value in results:
            idx = r_key.config_id + self.initial_run_num
            pipeline = self._api._backend.load_model_by_seed_and_id_and_budget(
                seed=self.seed,
                idx=idx,
                budget=r_key.budget,
            )
            pipeline = self._api._create_pipeline_from_representation(pipeline)
            model, kenning_conf = self.extract_model(
                pipeline,
            )
            self.model_paths.append(
                self.output_directory / f"{self.seed}_{idx}_{r_key.budget}.pth"
            )
            torch.save(model.state_dict(), self.model_paths[-1])
            kenning_conf["model_wrapper"]["parameters"]["model_path"] = str(
                self.model_paths[-1]
            )
            self.best_configs.append(kenning_conf)
            yield kenning_conf

    def pre_training_callback(self, data: Dict) -> Optional[float]:
        """
        Function called by AutoPyTorch right before training.

        Parameters
        ----------
        data : Dict
            The AutoPyTorch trainer data.

        Returns
        -------
        Optional[float]
            The model size after optimizations.

        Raises
        ------
        ModelTooLargeError
            If model size is too large to fit into the platform.
        """
        # Skip model size check if max size is not specified
        if self.max_model_size is None:
            return

        import torch
        from autoPyTorch.pipeline.components.training.trainer import (
            ModelTooLargeError,
        )

        from kenning.modelwrappers.frameworks.pytorch import PyTorchWrapper

        # Extract model and model wrapper class
        model: torch.nn.Module = data["network_backbone"]
        model_wrapper_class = None
        extracted_model = None
        for mw_class in self.use_models:
            extracted_model = mw_class.extract_model(model)
            if extracted_model is not None:
                model_wrapper_class = mw_class
                model = extracted_model
                break
        if model_wrapper_class is None:
            raise ModelExtractionError(
                f"Cannot extract Kenning model from: {model}"
            )

        with NamedTemporaryFile("w") as fd:
            # Save model to file
            model_wrapper: PyTorchWrapper = model_wrapper_class(
                fd.name, self.reduced_dataset
            )
            model_wrapper.model = model
            model_wrapper.model_prepared = True
            model_wrapper.save_model(fd.name)

            # Run Kenning optimize flow for the model
            if self.optimizers:
                runner = PipelineRunner(
                    dataset=self.reduced_dataset,
                    optimizers=self.optimizers,
                    platform=self.platform,
                    model_wrapper=model_wrapper,
                )
                try:
                    for opt in self.optimizers:
                        opt.model_wrapper = model_wrapper
                        opt.dataset = self.reduced_dataset
                    opt_model_path = runner._handle_optimizations()
                finally:
                    for opt in self.optimizers:
                        opt.model_wrapper = None
                        opt.dataset = self.dataset
                model_size = opt_model_path.stat().st_size / 1024
                # Cleanup optimized models
                for opt in self.optimizers:
                    opt.compiled_model_path.unlink()
            else:
                model_size = model_wrapper.get_model_size()

        # Validate model size
        if self.max_model_size < model_size:
            raise ModelTooLargeError(
                f"Model size ({model_size} KB) larger "
                f"than maximum ({self.max_model_size} KB)"
            )
        return model_size
