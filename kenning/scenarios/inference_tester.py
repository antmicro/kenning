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

import yaml
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
from kenning.cli.completers import (
    DATASETS,
    MODEL_WRAPPERS,
    OPTIMIZERS,
    PLATFORMS,
    RUNTIME_PROTOCOLS,
    RUNTIMES,
    ClassPathCompleter,
)
from kenning.core.measurements import MeasurementsCollector
from kenning.dataconverters.modelwrapper_dataconverter import (
    ModelWrapperDataConverter,
)
from kenning.utils.args_manager import ensure_exclusive_cfg_or_flags
from kenning.utils.class_loader import get_command, load_class
from kenning.utils.logger import KLogger
from kenning.utils.pipeline_runner import (
    PipelineRunner,
    PipelineRunnerInvalidConfigError,
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
        required_prefix = ""
        if TRAIN not in types:
            # 'train' is not used, JSON and flag configuration available
            groups = CommandTemplate.add_groups(parser, groups, ARGS_GROUPS)
            required_prefix = "* "
            groups[FILE_CONFIG].add_argument(
                "--json-cfg",
                "--cfg",
                help=f"{required_prefix}The path to the input JSON file with configuration of the inference",  # noqa: E501
                type=ResourceURI,
            ).completer = FilesCompleter(
                allowednames=("*.json", "*.yaml", "*.yml")
            )
            flag_group = groups[FLAG_CONFIG]
            shared_flags_group = flag_group
        else:
            # 'train' is not compatible with JSON configuration
            flag_group = parser.add_argument_group(GROUP_SCHEMA.format(TEST))
            shared_flags_group = other_group

        shared_flags_group.add_argument(
            "--platform-cls",
            help="Platform-based class that wraps platform being tested",
        ).completer = ClassPathCompleter(PLATFORMS)
        if include_modelwrapper:
            shared_flags_group.add_argument(
                "--modelwrapper-cls",
                help=f"{required_prefix}ModelWrapper-based class with inference implementation to import",  # noqa: E501
                required=TRAIN in types,
            ).completer = ClassPathCompleter(MODEL_WRAPPERS)
        dataset_flag = shared_flags_group.add_argument(
            "--dataset-cls",
            help="Dataset-based class with dataset to import",
            required=TRAIN in types,
        )
        dataset_flag.completer = ClassPathCompleter(DATASETS)
        # 'optimize' specific arguments
        if not types or OPTIMIZE in types:
            flag_group.add_argument(
                "--compiler-cls",
                help=f"{required_prefix}Optimizer-based class with compiling routines to import",  # noqa: E501
            ).completer = ClassPathCompleter(OPTIMIZERS)
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
        # 'test' specific arguments
        if not types or TEST in types:
            if include_measurements:
                other_group.add_argument(
                    "--measurements",
                    help="The path to the output JSON file with measurements",
                    nargs=1,
                    type=Path,
                    default=[None],
                    required=bool(types),
                )
            other_group.add_argument(
                "--evaluate-unoptimized",
                help="Test model before optimization and append measurements",
                action="store_true",
            )
            dataset_flag.help = f"{required_prefix}{dataset_flag.help}"
            flag_group.add_argument(
                "--runtime-cls",
                help="Runtime-based class with the implementation of model runtime",  # noqa: E501
            ).completer = ClassPathCompleter(RUNTIMES)
            flag_group.add_argument(
                "--protocol-cls",
                help="Protocol-based class with the implementation of communication between inference tester and inference runner",  # noqa: E501
            ).completer = ClassPathCompleter(RUNTIME_PROTOCOLS)
        # Only when scenario is used outside of Kenning CLI
        if not types:
            other_group.add_argument(
                "--run-benchmarks-only",
                help="Instead of running the full compilation and testing flow, only testing of the model is executed",  # noqa: E501
                action="store_true",
            )
        return parser, groups

    @staticmethod
    def prepare_args(
        args: argparse.Namespace, required_flags: List[str]
    ) -> argparse.Namespace:
        """
        Prepares and validates parased arguments.

        Parameters
        ----------
        args : argparse.Namespace
            Parsed arguments.
        required_flags : List[str]
            Flags required for this command.

        Returns
        -------
        argparse.Namespace
            Validated parsed arguments.
        """
        InferenceTester._fill_missing_namespace_args(args)
        InferenceTester._ensure_exclusive_cfg_or_flags(args, required_flags)
        return args

    @staticmethod
    def _fill_missing_namespace_args(args: argparse.Namespace):
        if "json_cfg" not in args:
            args.json_cfg = None
        if "measurements" not in args:
            args.measurements = [None]
        if "evaluate_unoptimized" not in args:
            args.evaluate_unoptimized = False

    @staticmethod
    def _ensure_exclusive_cfg_or_flags(
        args: argparse.Namespace, required_flags: List[str]
    ):
        required_args = (
            [1] + [2]
            if args.measurements[0] is not None
            else [] + [3]
            if "compiler_cls" in args
            else []
        )
        ensure_exclusive_cfg_or_flags(args, required_flags, required_args)

    @staticmethod
    def run(args: argparse.Namespace, not_parsed: List[str] = [], **kwargs):
        command = get_command()

        flag_config_args = [
            "platform_cls",
            "modelwrapper_cls",
            "dataset_cls",
            "compiler_cls",
            "runtime_cls",
            "protocol_cls",
        ]
        args = InferenceTester.prepare_args(args, flag_config_args)

        if args.help:
            raise ParserHelpException
        if args.json_cfg is not None:
            return InferenceTester._run_from_cfg(
                args, command, not_parsed=not_parsed, **kwargs
            )
        return InferenceTester._run_from_flags(
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
            json_cfg = yaml.safe_load(f)

        pipeline_runner = PipelineRunner.from_json_cfg(
            json_cfg,
            cfg_path=args.json_cfg,
        )

        return InferenceTester._run_pipeline(
            args=args, command=command, pipeline_runner=pipeline_runner
        )

    @staticmethod
    def _run_from_flags(
        args: argparse.Namespace,
        command: List[str],
        not_parsed: List[str] = [],
        **kwargs,
    ):
        platformcls = (
            load_class(args.platform_cls) if args.platform_cls else None
        )
        modelwrappercls = (
            load_class(args.modelwrapper_cls)
            if args.modelwrapper_cls
            else None
        )
        datasetcls = (
            load_class(args.dataset_cls)
            if getattr(args, "dataset_cls", None)
            else None
        )
        runtimecls = (
            load_class(args.runtime_cls)
            if getattr(args, "runtime_cls", None)
            else None
        )
        compilercls = (
            load_class(args.compiler_cls)
            if getattr(args, "compiler_cls", None)
            else None
        )
        protocolcls = (
            load_class(args.protocol_cls)
            if getattr(args, "protocol_cls", None)
            else None
        )

        if not compilercls and (protocolcls and not runtimecls):
            raise argparse.ArgumentError(
                None, "'--protocol-cls' requires '--runtime-cls' to be defined"
            )

        parser = argparse.ArgumentParser(
            " ".join(map(lambda x: x.strip(), get_command(with_slash=False)))
            + "\n",
            parents=[]
            + ([platformcls.form_argparse()[0]] if platformcls else [])
            + ([modelwrappercls.form_argparse()[0]] if modelwrappercls else [])
            + ([datasetcls.form_argparse()[0]] if datasetcls else [])
            + ([runtimecls.form_argparse()[0]] if runtimecls else [])
            + ([compilercls.form_argparse()[0]] if compilercls else [])
            + ([protocolcls.form_argparse()[0]] if protocolcls else []),
            add_help=False,
        )

        if args.help:
            raise ParserHelpException(parser)
        args = parser.parse_args(not_parsed, namespace=args)

        platform = platformcls.from_argparse(args) if platformcls else None
        dataset = datasetcls.from_argparse(args) if datasetcls else None
        model = (
            modelwrappercls.from_argparse(dataset, args)
            if modelwrappercls
            else None
        )
        optimizers = (
            [compilercls.from_argparse(dataset, args)] if compilercls else []
        )
        protocol = protocolcls.from_argparse(args) if protocolcls else None
        runtime = runtimecls.from_argparse(args) if runtimecls else None

        # TODO: This is a temporal solution, in future dataconverter
        # should be parsed separately
        dataconverter = ModelWrapperDataConverter(model)

        pipeline_runner = PipelineRunner(
            platform=platform,
            dataset=dataset,
            dataconverter=dataconverter,
            optimizers=optimizers,
            runtime=runtime,
            protocol=protocol,
            model_wrapper=model,
        )

        return InferenceTester._run_pipeline(
            args=args, command=command, pipeline_runner=pipeline_runner
        )

    @staticmethod
    def _run_pipeline(
        args: argparse.Namespace,
        command: List[str],
        pipeline_runner: PipelineRunner,
    ):
        from kenning.cli.config import get_used_subcommands

        subcommands = get_used_subcommands(args)
        output = args.measurements[0] if args.measurements[0] else None
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
                    raise PipelineRunnerInvalidConfigError(
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
            KLogger.error(f"Validation error: {ex}", stack_info=True)
            raise
        except Exception as ex:
            KLogger.error(ex, stack_info=True)
            raise

        if ret is None:
            return 1
        return ret


if __name__ == "__main__":
    sys.exit(InferenceTester.scenario_run())
