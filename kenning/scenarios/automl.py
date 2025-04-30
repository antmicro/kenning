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
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from kenning.cli.command_template import (
    OPTIMIZE,
    TEST,
    TRAIN,
    ArgumentsGroups,
    ParserHelpException,
)
from kenning.cli.completers import (
    AUTOML,
    DATASETS,
    OPTIMIZERS,
    PLATFORMS,
    RUNTIME_PROTOCOLS,
    RUNTIMES,
    ClassPathCompleter,
)
from kenning.core.automl import AutoML
from kenning.scenarios.inference_tester import (
    DEFAULT_GROUP,
    FLAG_CONFIG,
    InferenceTester,
)
from kenning.utils.automl_runner import AutoMLRunner
from kenning.utils.class_loader import (
    ConfigKey,
    get_command,
    load_class_by_type,
)
from kenning.utils.pipeline_runner import PipelineRunner


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

        required_prefix = "* "
        flag_group = groups[FLAG_CONFIG]
        flag_group.add_argument(
            "--automl-cls",
            help=f"{required_prefix}AutoML-based class with AutoML flow implementation",  # noqa: E501
        ).completer = ClassPathCompleter(AUTOML)

        other_group = groups[DEFAULT_GROUP]
        other_group.add_argument(
            "--allow-failures",
            help="Fail only if all generated scenarios failed",
            action="store_true",
        )

        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, not_parsed: List[str] = [], **kwargs):
        command = get_command()

        flag_config_names = [
            "automl_cls",
            "platform_cls",
            "dataset_cls",
            "compiler_cls",
            "runtime_cls",
            "protocol_cls",
        ]
        args = AutoMLCommand.prepare_args(args, flag_config_names)
        if args.json_cfg:
            if args.help:
                raise ParserHelpException
            return AutoMLCommand._run_from_cfg(
                args, command, not_parsed=not_parsed, **kwargs
            )
        return AutoMLCommand._run_from_flags(
            args, command, not_parsed=not_parsed, **kwargs
        )

    @staticmethod
    def _run_from_cfg(
        args: argparse.Namespace,
        command: List[str],
        not_parsed: List[str] = [],
        **kwargs,
    ):
        if not_parsed:
            raise argparse.ArgumentError(
                None, f"unrecognized arguments: {' '.join(not_parsed)}"
            )

        with open(args.json_cfg, "r") as f:
            cfg = yaml.safe_load(f)

        automl_runner = AutoMLRunner.from_json_cfg(cfg)
        return AutoMLCommand._run_pipeline(args, command, automl_runner)

    @staticmethod
    def _run_from_flags(
        args: argparse.Namespace,
        command: List[str],
        not_parsed: List[str] = [],
        **kwargs,
    ):
        automlcls = load_class_by_type(
            getattr(args, "automl_cls", None), AUTOML
        )
        platformcls = load_class_by_type(
            getattr(args, "platform_cls", None), PLATFORMS
        )
        datasetcls = load_class_by_type(
            getattr(args, "dataset_cls", None), DATASETS
        )
        runtimecls = load_class_by_type(
            getattr(args, "runtime_cls", None), RUNTIMES
        )
        compilercls = load_class_by_type(
            getattr(args, "compiler_cls", None), OPTIMIZERS
        )
        protocolcls = load_class_by_type(
            getattr(args, "protocol_cls", None), RUNTIME_PROTOCOLS
        )

        if not compilercls and (protocolcls and not runtimecls):
            raise argparse.ArgumentError(
                None, "'--protocol-cls' requires '--runtime-cls' to be defined"
            )

        parser = argparse.ArgumentParser(
            " ".join(map(lambda x: x.strip(), get_command(with_slash=False)))
            + "\n",
            parents=[
                cls.form_argparse(args)[0]
                for cls in (
                    automlcls,
                    datasetcls,
                    runtimecls,
                    compilercls,
                    protocolcls,
                    platformcls,
                )
                if cls
            ],
            add_help=False,
        )

        if args.help:
            raise ParserHelpException(parser)
        args = parser.parse_args(not_parsed, namespace=args)

        platform = platformcls.from_argparse(args) if platformcls else None
        dataset = datasetcls.from_argparse(args) if datasetcls else None
        automl = (
            automlcls.from_argparse(dataset, platform, args)
            if automlcls
            else None
        )
        optimizers = (
            [compilercls.from_argparse(dataset, args)] if compilercls else []
        )
        protocol = protocolcls.from_argparse(args) if protocolcls else None
        runtime = runtimecls.from_argparse(args) if runtimecls else None

        conf = {
            key.name: obj.to_json()
            for key, obj in (
                (ConfigKey.platform, platform),
                (ConfigKey.dataset, dataset),
                (ConfigKey.automl, automl),
                (ConfigKey.runtime, runtime),
                (ConfigKey.protocol, protocol),
            )
            if obj
        }
        conf[ConfigKey.optimizers.name] = [opt.to_json() for opt in optimizers]
        automl_runner = AutoMLRunner(
            dataset=dataset,
            autoML=automl,
            pipeline_config=conf,
        )
        return AutoMLCommand._run_pipeline(args, command, automl_runner)

    @staticmethod
    def _run_pipeline(
        args: argparse.Namespace,
        command: List[str],
        automl_runner: AutoMLRunner,
    ):
        from kenning.cli.config import get_used_subcommands

        best_configs = automl_runner.run(
            args.verbosity,
        )

        subcommands = get_used_subcommands(args)
        measurements = []
        model_names = []
        rets = []
        run_pipeline = bool({OPTIMIZE, TEST}.intersection(subcommands))
        n_valid_models = 0
        for path, conf in best_configs:
            model_path = Path(
                conf[ConfigKey.model_wrapper.name]["parameters"]["model_path"]
            )
            if TEST in subcommands:
                measurements.append(
                    str(model_path.with_suffix(".measurements.json"))
                )
                model_names.append(path.stem)
                args.measurements = [measurements[-1]]
            # Run InferenceTester flow - optimization and evaluation
            if run_pipeline:
                pipeline_runner = PipelineRunner.from_json_cfg(
                    conf,
                    cfg_path=path,
                )
                try:
                    ret = InferenceTester._run_pipeline(
                        args, command, pipeline_runner
                    )
                except Exception:
                    ret = 1
                    measurements.pop(-1)
                    model_names.pop(-1)
                else:
                    n_valid_models += 1
                rets.append(ret)
                if n_valid_models >= automl_runner.autoML.n_best_models:
                    break

        # Set all available measurement for comparison report
        args.measurements = measurements
        args.automl_stats = (
            automl_runner.autoML.output_directory / AutoML.STATS_FILE_NAME
        )
        args.model_names = model_names
        if not run_pipeline:
            return 0
        if len(rets) == 0:
            return 1
        if args.allow_failures:
            return 1 if all(rets) else 0
        return 1 if any(rets) else 0


if __name__ == "__main__":
    sys.exit(AutoMLCommand.scenario_run())
