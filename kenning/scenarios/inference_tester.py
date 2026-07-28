#!/usr/bin/env python

# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A script that runs inference client.

It requires implementations of two classes as input:

* ModelWrapper - wraps the model that will be compiled and executed on hardware
* Optimizer - wraps the compiling routines for the deep learning model

Three classes are optional. Not every combination is a valid configuration:
* Protocol - describes the protocol over which the communication is
  performed
* Dataset - provides data for benchmarking
* Runtime - provides a runtime to run the model

If Runtime is not provided then providing either Optimizer or Protocol
raises an Exception, as this is not a valid scenario.

If Protocol is specified then it is expected that an instance of an
inference server is running. Otherwise the inference is run locally.

If Runtime is not specified then a native framework of the model is used to
run the inference. Otherwise the provided Runtime is used.

If Optimizer is not specified, then the script runs the input model either
using provided Runtime or in its native framework. Otherwise the Optimizer
compiles the model before passing it to the Runtime.

Each of those classes require specific set or arguments to configure the
compilation and benchmark process.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from argcomplete.completers import FilesCompleter
from jsonschema.exceptions import ValidationError

from kenning.cli.command_template import (
    DEFAULT_GROUP,
    GROUP_SCHEMA,
    OPTIMIZE,
    REPORT,
    TEST,
    TRAIN,
    ArgumentsGroups,
    CommandTemplate,
    ParserHelpException,
    generate_command_type,
)
from kenning.cli.parser import get_used_subcommands
from kenning.core.measurements import MeasurementsCollector
from kenning.dispatcher.block_config import (
    set_block_direct_argument,
)
from kenning.utils.class_loader import (
    ConfigKey,
    get_command,
    objs_from_full_dict_config,
)
from kenning.utils.logger import KLogger
from kenning.utils.pipeline_runner import (
    PipelineRunner,
)
from kenning.utils.resource_manager import ResourceURI

FILE_CONFIG = "Inference configuration with JSON/YAML file"
FLAG_CONFIG = "Inference configuration with flags"
ARGS_GROUPS = {
    FILE_CONFIG: f"Configuration with pipeline defined in JSON/YAML file. This section is not compatible with '{FLAG_CONFIG}'. Arguments with '*' are required.",  # noqa: E501
    FLAG_CONFIG: f"Configuration with flags. This section is not compatible with '{FILE_CONFIG}'. Arguments with '*' are required.",  # noqa: E501
}


class InferenceTester(CommandTemplate):
    """
    Command template for running inference benchmarking.
    """

    parse_all = False
    description = {
        TEST: "    A script that runs inference and gathers measurements.",
        OPTIMIZE: "    A script that optimize model.",
    }
    ID = generate_command_type()

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Optional[ArgumentsGroups] = None,
        include_modelwrapper: bool = True,
        include_measurements: bool = True,
    ) -> Tuple[argparse.ArgumentParser, ArgumentsGroups]:
        parser, groups = super(
            InferenceTester, InferenceTester
        ).configure_parser(
            parser,
            command,
            types,
            groups,
            (len(types) > 1 and REPORT in types) or TRAIN in types,
        )

        other_group = groups[DEFAULT_GROUP]
        groups = CommandTemplate.add_groups(parser, groups, ARGS_GROUPS)
        groups[FILE_CONFIG].add_argument(
            "--json-cfg",
            "--cfg",
            help="The path to the input JSON file with configuration of the inference",  # noqa: E501
            type=ResourceURI,
        ).completer = FilesCompleter(
            allowednames=("*.json", "*.yaml", "*.yml")
        )

        if TRAIN not in types:
            flag_group = groups[FLAG_CONFIG]
            shared_flags_group = flag_group
        else:
            flag_group = parser.add_argument_group(GROUP_SCHEMA.format(TEST))
            shared_flags_group = other_group

        block_types_for_shared_group = [ConfigKey.platform, ConfigKey.dataset]
        if include_modelwrapper:
            block_types_for_shared_group.append(ConfigKey.model_wrapper)

        CommandTemplate.add_block_flags_to_argparse(
            shared_flags_group, block_types_for_shared_group
        )

        block_types = []
        # 'optimize' specific arguments
        if not types or OPTIMIZE in types:
            block_types.append(ConfigKey.optimizers)
            other_group.add_argument(
                "--convert-to-onnx",
                help="Before compiling the model, convert it to ONNX and use in compilation (provide a path to save here)",  # noqa: E501
                type=Path,
            )
            other_group.add_argument(
                "--max-target-side-optimizers",
                help="Max number of consecutive target-side optimizers",
                type=int,
                default=-1,
            )

        if include_measurements:
            block_types.append(ConfigKey.report)

        # 'test' specific arguments
        if not types or TEST in types:
            other_group.add_argument(
                "--evaluate-unoptimized",
                help="Test model before optimization and append measurements",
                action="store_true",
            )
            block_types += [ConfigKey.runtime, ConfigKey.protocol]
        CommandTemplate.add_block_flags_to_argparse(flag_group, block_types)

        # Only when scenario is used outside of Kenning CLI
        if not types:
            other_group.add_argument(
                "--run-benchmarks-only",
                help="Instead of running the full compilation and testing flow, only testing of the model is executed",  # noqa: E501
                action="store_true",
            )
        return parser, groups

    @staticmethod
    def prepare_args(args: argparse.Namespace) -> argparse.Namespace:
        """
        Prepares and validates parased arguments.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed arguments.

        Returns
        -------
        argparse.Namespace
            Validated parsed arguments.
        """
        InferenceTester._fill_missing_namespace_args(args)
        return args

    @staticmethod
    def _fill_missing_namespace_args(args: argparse.Namespace):
        if "json_cfg" not in args:
            args.json_cfg = None
        if "evaluate_unoptimized" not in args:
            args.evaluate_unoptimized = False

    @staticmethod
    def run(args: argparse.Namespace, not_parsed: List[str] = [], **kwargs):
        command = get_command()
        if args.help:
            raise ParserHelpException
        keys = [
            ConfigKey.platform,
            ConfigKey.model_wrapper,
            ConfigKey.dataset,
            ConfigKey.runtime,
            ConfigKey.optimizers,
            ConfigKey.protocol,
            ConfigKey.report,
            ConfigKey.runtime_builder,
            ConfigKey.inference_loop,
        ]

        if REPORT not in get_used_subcommands(args):
            InferenceTester.default_block_classes[
                ConfigKey.report
            ] = "StubReport"

        config = InferenceTester.parse_configuration(args, not_parsed, keys)

        config = set_block_direct_argument(
            "from_file", True, config, ConfigKey.model_wrapper
        )

        objs = objs_from_full_dict_config(config)

        if ConfigKey.report in objs.keys():
            args.parsed_report = objs[ConfigKey.report]

            report_type = type(objs[ConfigKey.report]).__name__

            KLogger.debug(f"Selected report type: {report_type}")

        pipeline_runner = PipelineRunner.from_objs_dict(objs)

        return InferenceTester._run_pipeline(
            args=args, command=command, pipeline_runner=pipeline_runner
        )

    @staticmethod
    def _run_pipeline(
        args: argparse.Namespace,
        command: List[str],
        pipeline_runner: PipelineRunner,
    ):
        subcommands = get_used_subcommands(args)

        output = None

        # this is added to make inference testser's tests work
        if pipeline_runner.output is None:
            output = (
                args.measurements[0]
                if getattr(args, "measurements", None) is not None
                else None
            )
        else:
            output = pipeline_runner.output

        verbosity = args.verbosity
        convert_to_onnx = getattr(args, "convert_to_onnx", False)
        max_target_side_optimizers = getattr(
            args, "max_target_side_optimizers", -1
        )
        run_optimizations = (
            OPTIMIZE in subcommands
            and not getattr(args, "run_benchmarks_only", False)
            and len(pipeline_runner.optimizers) > 0
        )
        run_benchmarks = (
            TEST in subcommands and pipeline_runner.dataset is not None
        )
        try:
            ret = pipeline_runner.run(
                output=output,
                verbosity=verbosity,
                convert_to_onnx=convert_to_onnx,
                max_target_side_optimizers=max_target_side_optimizers,
                command=command,
                run_optimizations=run_optimizations,
                run_benchmarks=run_benchmarks,
            )

            evaluate_unoptimized = getattr(args, "evaluate_unoptimized", False)
            if evaluate_unoptimized and not ret and output:
                if not run_optimizations:
                    raise ValueError(
                        "If optimizations are skipped, the model will already "
                        "be unoptimized, thus '--evaluate-unoptimized' is "
                        "redundant"
                    )
                unoptimized_output = output.parent / (
                    "unoptmized_" + output.name
                )
                pipeline_runner.optimizers = []
                ret |= pipeline_runner.run(
                    output=unoptimized_output,
                    verbosity=verbosity,
                    convert_to_onnx=convert_to_onnx,
                    command=command,
                    run_optimizations=False,
                    run_benchmarks=run_benchmarks,
                )
                MeasurementsCollector.set_unoptimized(
                    output, unoptimized_output
                )
        except ValidationError as ex:
            KLogger.error(
                f"Validation error: {ex}", exc_info=ex, stack_info=True
            )
            raise

        if ret is None:
            return 1
        return ret

    @staticmethod
    def get_overridable(subcommands: List[str]) -> List[ConfigKey]:
        if TRAIN in subcommands:
            return []

        classes = [
            ConfigKey.platform,
            ConfigKey.model_wrapper,
            ConfigKey.dataset,
        ]

        if TEST in subcommands:
            classes.extend(
                [
                    ConfigKey.runtime,
                    ConfigKey.protocol,
                ]
            )

        return classes


if __name__ == "__main__":
    sys.exit(InferenceTester.scenario_run())
