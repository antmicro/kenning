# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A script that runs an AutoML flow:
* looking for the best models for a given dataset,
* optimizing found models,
* evaluating the models.

It requires implementations of two classes as input:
* AutoML - wraps the AutoML framework
* Dataset - provides data for training and evaluating models.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml

from kenning.cli.command_template import (
    OPTIMIZE,
    REPORT,
    TEST,
    TRAIN,
    ArgumentsGroups,
)
from kenning.cli.completers import (
    AUTOML,
)
from kenning.cli.parser import get_used_subcommands
from kenning.core.automl import AutoML
from kenning.dispatcher.block_config import (
    filter_block_types_from_config_dict,
    merge_config_dicts,
    yaml_or_json_to_config_dict,
)
from kenning.scenarios.inference_tester import (
    DEFAULT_GROUP,
    FLAG_CONFIG,
    InferenceTester,
)
from kenning.utils.automl_runner import AutoMLRunner
from kenning.utils.class_loader import (
    ConfigKey,
    get_command,
    objs_from_full_dict_config,
)
from kenning.utils.logger import KLogger
from kenning.utils.pipeline_runner import PipelineRunner


class AutoMLCache:
    """
    AutoML cache management class.
    """

    def _condition_run(func):
        """
        Conditions method execution based on AutoMLCache.cache_path presents.
        """

        def wrapper(*args, **kwargs):
            if AutoMLCache.cache_path is not None:
                return func(*args, **kwargs)

        return wrapper

    cache_path: Union[Path, None] = None

    @_condition_run
    @staticmethod
    def ensure_created():
        """
        Creates cache directory or does nothing if it already exists.
        """
        if not AutoMLCache.cache_path.exists():
            AutoMLCache.cache_path.mkdir(exist_ok=True)

    @_condition_run
    @staticmethod
    def clean():
        """
        Walks through all files in cache directory and deets deletes them.
        """
        for file in AutoMLCache.files():
            AutoMLCache.delete(file)

    @_condition_run
    @staticmethod
    def files():
        """
        Yields cached file paths for easy iteration.
        """
        for root, dirs, files in os.walk(
            str(AutoMLCache.cache_path), topdown=False
        ):
            for name in files:
                yield Path(root) / name

    @_condition_run
    @staticmethod
    def save(source_path):
        """
        Caches file by creating a symlink to given 'source_path'.
        """
        try:
            (AutoMLCache.cache_path / source_path.name).symlink_to(
                source_path.resolve()
            )
        except (NotImplementedError, FileNotFoundError) as ex:
            KLogger.warning(
                f"Unable to save item into automl cache. Error: {ex}"
            )

    @_condition_run
    @staticmethod
    def delete(path):
        """
        Deletes file from the cache directory.
        """
        try:
            (AutoMLCache.cache_path / path.resolve().name).unlink()
        except FileNotFoundError as ex:
            KLogger.warning(
                f"Unable to remove item from automl cache. Error: {ex}"
            )


class AutoMLCommand(InferenceTester):
    """
    Command template for running AutoML flow.
    """

    parse_all = False
    description = {
        AUTOML: "    An AutoML flow search the best model for a given dataset.",  # noqa: E501
        OPTIMIZE: InferenceTester.description[OPTIMIZE],
        TEST: InferenceTester.description[TEST],
    }
    ID = InferenceTester.ID

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Optional[ArgumentsGroups] = None,
    ) -> Tuple[argparse.ArgumentParser, ArgumentsGroups]:
        assert (
            TRAIN not in types
        ), "AutoML has training embedded, does not support `train` subcommand"
        parser, groups = super(AutoMLCommand, AutoMLCommand).configure_parser(
            parser,
            command,
            types,
            groups,
            include_modelwrapper=False,
            include_measurements=False,
        )

        flag_group = groups[FLAG_CONFIG]
        InferenceTester.add_block_flags_to_argparse(
            flag_group, [ConfigKey.automl]
        )

        other_group = groups[DEFAULT_GROUP]
        other_group.add_argument(
            "--allow-failures",
            help="Fail only if all generated scenarios failed",
            action="store_true",
        )

        # Exclude the flag for a case without further subcommands
        if len(types) != 1:
            other_group.add_argument(
                "--use-previous-results",
                help="Provide necessary resources for automl optimization/evaluation/report preparation based on the previous latest automl results and therefore skip automl search",  # noqa: E501
                action="store_true",
            )

        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, not_parsed: List[str] = [], **kwargs):
        command = get_command()

        keys = [
            ConfigKey.automl,
            ConfigKey.platform,
            ConfigKey.dataset,
            ConfigKey.runtime,
            ConfigKey.optimizers,
            ConfigKey.protocol,
            ConfigKey.runtime_builder,
        ]

        config = AutoMLCommand.parse_configuration(args, not_parsed, keys)

        return AutoMLCommand._run_pipeline(
            args,
            command,
            config,
        )

    @staticmethod
    def _run_pipeline(
        args: argparse.Namespace,
        command: List[str],
        initial_config: Dict,
    ):
        objs = objs_from_full_dict_config(initial_config)

        conf = {
            key.name: obj.to_json()
            for key in (
                ConfigKey.automl,
                ConfigKey.platform,
                ConfigKey.dataset,
                ConfigKey.runtime,
                ConfigKey.protocol,
            )
            if (obj := objs.get(key))
        }
        conf[ConfigKey.optimizers.name] = [
            opt.to_json() for opt in objs[ConfigKey.optimizers]
        ]

        automl_runner = AutoMLRunner.from_objs_dict(objs, pipeline_config=conf)

        subcommands = get_used_subcommands(args)
        measurements = []
        model_names = []
        rets = []
        run_pipeline = bool({OPTIMIZE, TEST}.intersection(subcommands))
        use_previous_results = (
            args.use_previous_results if len(subcommands) > 1 else None
        )
        n_valid_models = 0

        run_benchmarks = TEST in subcommands
        run_optimizations = None

        AutoMLCache.cache_path = (
            automl_runner.autoML.output_directory / ".cache"
        ).resolve()

        AutoMLCache.ensure_created()

        if use_previous_results:
            paths_cfgs = []
            for file in AutoMLCache.files():
                if str(file.name).startswith("automl_conf_"):
                    with open(file, "r") as f:
                        cfg = yaml.safe_load(f)
                    paths_cfgs.append((AutoMLCache.cache_path / file, cfg))

            def paths_cfgs_provider(*arg):
                yield from paths_cfgs

            automl_runner.run = paths_cfgs_provider
        else:
            AutoMLCache.clean()

        # Run method can be overridden by cache
        # and return results from previous runs
        best_configs = automl_runner.run(
            args.verbosity,
        )

        # Manage automl cache dir
        if use_previous_results and run_benchmarks:
            for file in AutoMLCache.files():
                if "measurements" in str(file):
                    AutoMLCache.delete(file)

        for path, conf in best_configs:
            if not use_previous_results:
                AutoMLCache.save(path)

            model_path = Path(
                conf[ConfigKey.model_wrapper.name]["parameters"]["model_path"]
            )
            if run_benchmarks or REPORT in subcommands:
                model_names.append(path.stem)
                if run_benchmarks:
                    measurements.append(
                        str(model_path.with_suffix(".measurements.json"))
                    )
                    args.measurements = [measurements[-1]]

            # Run InferenceTester flow - optimization and evaluation
            if run_pipeline:
                # AutoML runner returns pipeline config in the same format as
                # Kenning's yaml/dict config files. We need to convert it to a
                # standard config.
                conf = yaml_or_json_to_config_dict(conf)
                conf = merge_config_dicts(initial_config, conf)
                conf = filter_block_types_from_config_dict(
                    [
                        ConfigKey.model_wrapper,
                        ConfigKey.dataset,
                        ConfigKey.optimizers,
                        ConfigKey.runtime,
                        ConfigKey.platform,
                        ConfigKey.runtime_builder,
                        ConfigKey.protocol,
                    ],
                    conf,
                )
                pipeline_runner = PipelineRunner.from_objs_dict(
                    objs_from_full_dict_config(conf)
                )

                run_optimizations = (
                    OPTIMIZE in subcommands
                    and len(pipeline_runner.optimizers) > 0
                )

                try:
                    ret = InferenceTester._run_pipeline(
                        args, command, pipeline_runner
                    )

                    # Manage automl cache dir
                    if run_benchmarks:
                        measurements_path = Path(
                            args.measurements[-1]
                        ).resolve()
                        AutoMLCache.save(measurements_path)

                    test_output = (
                        args.measurements[-1]
                        if hasattr(args, "measurements")
                        and args.measurements[-1]
                        else None
                    )
                    evaluate_unoptimized = getattr(
                        args, "evaluate_unoptimized", False
                    )
                    if (
                        evaluate_unoptimized
                        and not ret
                        and test_output
                        and run_optimizations
                    ):
                        unoptimized_output = test_output.parent / (
                            "unoptmized_" + test_output.name
                        )
                        AutoMLCache.save(unoptimized_output)

                except Exception:
                    ret = 1
                    measurements.pop(-1)
                    model_names.pop(-1)
                else:
                    n_valid_models += 1
                rets.append(ret)
                if n_valid_models >= automl_runner.autoML.n_best_models:
                    break

        if not use_previous_results:
            AutoMLCache.save(
                automl_runner.autoML.output_directory / AutoML.STATS_FILE_NAME
            )

        # In case of 'kenning automl report ...'
        if use_previous_results and REPORT in subcommands:
            if not measurements:
                for file in AutoMLCache.files():
                    if "measurements" in file.name:
                        measurements.append(file)

            if not measurements:
                raise argparse.ArgumentError(
                    None,
                    "'report' with '--use-previous-results' used, but no measurements found in cache.",  # noqa: E501
                )

        # Set all available measurement for comparison report
        args.measurements = measurements
        args.automl_stats = (
            AutoMLCache.cache_path / AutoML.STATS_FILE_NAME
            if use_previous_results
            else automl_runner.autoML.output_directory / AutoML.STATS_FILE_NAME
        )
        args.model_names = model_names
        if not run_pipeline:
            return 0
        if len(rets) == 0:
            return 1
        if args.allow_failures:
            return 1 if all(rets) else 0
        return 1 if any(rets) else 0

    @staticmethod
    def get_overridable(subcommands: List[str]) -> List[ConfigKey]:
        overridable = super(AutoMLCommand, AutoMLCommand).get_overridable(
            subcommands
        )
        overridable.remove(ConfigKey.model_wrapper)
        overridable.append(ConfigKey.automl)

        return overridable


if __name__ == "__main__":
    sys.exit(AutoMLCommand.scenario_run())
