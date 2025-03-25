# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides an API for AutoML flow with AutoPyTorch framework.
"""

from multiprocessing import cpu_count
from pathlib import Path
from shutil import rmtree
from typing import (
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
import pandas as pd
import psutil
from sklearn.pipeline import Pipeline

from kenning.core.automl import AutoML, AutoMLModel
from kenning.core.dataset import Dataset
from kenning.core.platform import Platform
from kenning.utils.args_manager import get_type
from kenning.utils.logger import KLogger

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

    @staticmethod
    def define_forbidden_clauses(
        cs: ConfigurationSpace,
        **kwargs: Dict,
    ) -> ConfigurationSpace:
        """
        Defines forbidden clauses for not compatible pairs of hyperparametrs.

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

        Raises
        ------
        NotImplementedError
            If method has not been implemented in child class.
        """
        raise NotImplementedError

    @classmethod
    def build_backbone(
        cls,
        self: NetworkBackboneComponent,
        input_shape: Tuple[int, ...],
        dataset: Optional[Dataset] = None,
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
            Dataset used for the model.

        Returns
        -------
        PyTorchModel
            Created PyTorch model.

        Raises
        ------
        MissingConfigForAutoPyTorchModel
            If config requaries for model is missing.
        """
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
        return cls._create_model_structure(
            **args, input_shape=input_shape, dataset=dataset
        )

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
    def register_components(
        cls,
        dataset: Dataset,
    ) -> List[Type[NetworkBackboneComponent]]:
        """
        Dynamically creates new AutoPyTorch component classes,
        based on NetworkBackboneComponent, and register them.

        Parameters
        ----------
        dataset : Dataset
            Dataset used for the model.

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
        for module in network.modules():
            if type(module).__name__ == cls.model_class:
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
            if c_type is not list:
                model_wrapper_params[name] = backbone_conf[name]
            else:
                model_wrapper_params[name] = [
                    backbone_conf[f"{name}_{i}"]
                    for i in range(backbone_conf[name])
                ]
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
    }

    def __init__(
        self,
        dataset: Dataset,
        platform: Platform,
        output_directory: Path,
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
        """
        super().__init__(
            dataset=dataset,
            platform=platform,
            output_directory=output_directory,
            use_models=use_models if use_models else self.supported_models,
            time_limit=time_limit,
            optimize_metric=optimize_metric,
            n_best_models=n_best_models,
            seed=seed,
        )
        assert all(
            [issubclass(model, AutoPyTorchModel) for model in self.use_models]
        ), (
            "All provided classes in `use_models` have to "
            f"inherit from {AutoPyTorchModel.__name__}"
        )
        assert (
            budget_type in BUDGET_TYPES
        ), f"Budget has to be one of {BUDGET_TYPES}"

        self.use_models: List[AutoPyTorchModel]
        self.max_memory_usage = max_memory_usage
        self.max_evaluation_time = max_evaluation_time
        if max_evaluation_time is None:
            self.max_evaluation_time = self.time_limit
        self.budget_type = budget_type
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.application_size = application_size

        self.platform_ram = getattr(self.platform, "ram_size_kb", None)
        self.max_model_size = None
        if self.application_size and self.platform_ram:
            assert (
                self.platform_ram > self.application_size
            ), f"Application ({self.application_size}) does not fit into the platform ({self.platform_ram})"  # noqa: E501
            self.max_model_size = self.platform_ram - self.application_size
            self.max_model_size *= 0.95

        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self._components: List[Type[NetworkBackboneComponent]] = []
        self._prepared = False
        self._api = None

        self.initial_run_num: int = None
        self.model_paths: List[Path] = []
        self.best_configs: List[Path] = []

    def _preprocess_input(self, X: List[Iterable]) -> List[np.array]:
        """
        Flattens the input preserving a batch size.

        Parameters
        ----------
        X : List[Iterable]
            Input data.

        Returns
        -------
        List[np.array]
            Preprocessed input data.
        """
        X = [np.asarray(x) for x in X]
        return [x.reshape(x.shape[0], -1) for x in X]

    def prepare_framework(self):
        # Prepare dataset
        X_train, y_train = [], []
        for X, y in self.dataset.iter_train():
            X_train += self._preprocess_input(X)
            y_train += y[0]
        X_test, y_test = [], []
        for X, y in self.dataset.iter_test():
            X_test += self._preprocess_input(X)
            y_test += y[0]
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.asarray(y_train)
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.asarray(y_test)

        self.X_train = pd.DataFrame(
            X_train.reshape(X_train.shape[0], -1), dtype=np.float32
        )
        self.y_train = pd.DataFrame(y_train, dtype=np.int16)
        self.X_test = pd.DataFrame(
            X_test.reshape(X_test.shape[0], -1), dtype=np.float32
        )
        self.y_test = pd.DataFrame(y_test, dtype=np.int16)

        # Register models components
        for model in self.use_models:
            self._components += model.register_components(self.dataset)

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
            device="cpu",
            torch_num_threads=cpu_count(),
            use_tensorboard_logger=True,
            early_stopping=True,
        )
        if self.max_model_size:
            KLogger.info(f"Restricting model size to {self.max_model_size} KB")
            self._api.set_pipeline_options(
                max_model_size_kb=self.max_model_size,
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
