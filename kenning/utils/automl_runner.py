# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides runner for AutoML flow.
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Tuple

import yaml

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper

from kenning.core.automl import AutoML
from kenning.core.dataset import Dataset
from kenning.utils.class_loader import ConfigKey, obj_from_json
from kenning.utils.logger import KLogger


class AutoMLRunner(object):
    """
    Class responsible for running AutoML flow.
    """

    def __init__(
        self,
        dataset: Dataset,
        autoML: AutoML,
        pipeline_config: Dict,
    ):
        """
        Initializes the AutoMLRunner object.

        Parameters
        ----------
        dataset : Dataset
            Dataset object that provides data for training and inference.
        autoML : AutoML
            Definition of AutoML flow for chosen framework.
        pipeline_config : Dict
            Full pipeline configuration.
        """
        self.dataset = dataset
        self.autoML = autoML
        self.pipeline_config = pipeline_config

    @classmethod
    def from_json_cfg(cls, cfg: Dict):
        dataset = obj_from_json(cfg, ConfigKey.dataset)
        platform = obj_from_json(cfg, ConfigKey.platform)
        autoML = obj_from_json(
            cfg,
            ConfigKey.automl,
            dataset=dataset,
            platform=platform,
        )

        return cls(
            dataset=dataset,
            autoML=autoML,
            pipeline_config=cfg,
        )

    def run(self, verbosity: str = "INFO") -> Iterable[Tuple[Path, Dict]]:
        """
        Runs AutoML flow and returns the best found configurations.

        Configurations are also modified to contain unique paths,
        and saved to the output directory defined
        in AutoML object.

        Parameters
        ----------
        verbosity : str
            Verbosity level.

        Yields
        ------
        Tuple[Path, Dict]
            Dictionary with path to the saved configuration as a key
            and configuration as a value.
        """
        KLogger.debug("Preparing AutoML framework")
        self.autoML.prepare_framework()
        KLogger.debug("Running AutoML search")
        self.autoML.search()
        KLogger.info(str(self.autoML._api.sprint_statistics()))

        out_dir = self.autoML.output_directory

        for i, conf in enumerate(self.autoML.get_best_configs()):
            conf = deepcopy(self.pipeline_config | conf)
            model_wrapper_path = conf["model_wrapper"]["parameters"][
                "model_path"
            ]
            # Make paths unique for optimization and inference
            for optimizer in conf.get("optimizers", []):
                opt_parameters = optimizer.get("parameters", {})
                model_path = opt_parameters.get("compiled_model_path", None)
                if model_path is None:
                    model_path = model_wrapper_path.with_suffix(
                        f"{model_wrapper_path.suffix}.opt"
                    )
                model_path = Path(model_path)
                opt_parameters["compiled_model_path"] = str(
                    model_path.with_suffix(f".{i}{model_path.suffix}")
                )
                optimizer["parameters"] = opt_parameters
            if runtime := conf.get("runtime", None):
                run_params = runtime.get("parameters", {})
                run_params["save_model_path"] = opt_parameters[
                    "compiled_model_path"
                ]
                runtime["parameters"] = run_params
            conf_path = out_dir / f"automl_conf_{i}.yml"
            with conf_path.open("w") as fd:
                yaml.dump(conf, fd, Dumper=Dumper)
            yield conf_path, conf
