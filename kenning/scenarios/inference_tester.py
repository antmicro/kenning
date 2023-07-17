#!/usr/bin/env python

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A script that runs inference client.

It requires implementations of two classes as input:

* ModelWrapper - wraps the model that will be compiled and executed on hardware
* Optimizer - wraps the compiling routines for the deep learning model

Three classes are optional. Not every combination is a valid configuration:
* RuntimeProtocol - describes the protocol over which the communication is
  performed
* Dataset - provides data for benchmarking
* Runtime - provides a runtime to run the model

If Runtime is not provided then providing either Optimizer or RuntimeProtocol
raises an Exception, as this is not a valid scenario.

If RuntimeProtocol is specified then it is expected that an instance of an
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
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from jsonschema.exceptions import ValidationError

from kenning.cli.command_template import (
    CommandTemplate, TEST, OPTIMIZE, REPORT, DEFAULT_GROUP)
from kenning.utils.class_loader import get_command, load_class
from kenning.utils.pipeline_runner import run_pipeline, run_pipeline_json
import kenning.utils.logger as logger


JSON_CONFIG = "Inference configuration with JSON"
FLAG_CONFIG = "Inference configuration with flags"
ARGS_GROUPS = {
    JSON_CONFIG: f"Configuration with pipeline defined in JSON file. This section is not compatible with '{FLAG_CONFIG}'. Arguments with '*' are required.",  # noqa: E501
    FLAG_CONFIG: f"Configuration with flags. This section is not compatible with '{JSON_CONFIG}'. Arguments with '*' are required.",  # noqa: E501
}


class InferenceTester(CommandTemplate):
    parse_all = False
    description = {
        TEST: "A script that runs inference and gathers measurements.",
        OPTIMIZE: "A script that optimize model.",
    }

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Dict[str, argparse._ArgumentGroup] = None,
    ) -> Tuple[argparse.ArgumentParser, Dict]:
        parser, groups = super(
            InferenceTester, InferenceTester
        ).configure_parser(
            parser, command, types, groups, len(types) > 1 and REPORT in types)

        other_group = groups[DEFAULT_GROUP]
        for group_name in (JSON_CONFIG, FLAG_CONFIG):
            if group_name not in groups:
                groups[group_name] = parser.add_argument_group(
                    group_name, ARGS_GROUPS[group_name]
                )
        json_group = groups[JSON_CONFIG]
        flag_group = groups[FLAG_CONFIG]

        json_group.add_argument(
            '--json-cfg',
            help='* The path to the input JSON file with configuration of the inference',  # noqa: E501
        )
        flag_group .add_argument(
            '--modelwrapper-cls',
            help='* ModelWrapper-based class with inference implementation to import',  # noqa: E501
        )
        flag_group.add_argument(
            '--dataset-cls',
            help='* Dataset-based class with dataset to import',
        )
        if not types or OPTIMIZE in types:
            flag_group.add_argument(
                '--compiler-cls',
                help='* Optimizer-based class with compiling routines to import',  # noqa: E501
            )
            other_group.add_argument(
                '--convert-to-onnx',
                help='Before compiling the model, convert it to ONNX and use in compilation (provide a path to save here)',  # noqa: E501
                type=Path
            )
        if not types or TEST in types:
            other_group.add_argument(
                '--measurements',
                help='The path to the output JSON file with measurements',
                nargs=1,
                type=Path,
                required=True,
            )
            flag_group.add_argument(
                '--runtime-cls',
                help='Runtime-based class with the implementation of model runtime',  # noqa: E501
            )
            flag_group.add_argument(
                '--protocol-cls',
                help='RuntimeProtocol-based class with the implementation of communication between inference tester and inference runner',  # noqa: E501
            )
        if not types:
            other_group.add_argument(
                '--run-benchmarks-only',
                help='Instead of running the full compilation and testing flow, only testing of the model is executed',  # noqa: E501
                action='store_true'
            )
        return parser, groups

    @staticmethod
    def run(
        args: argparse.Namespace,
        not_parsed: List[str] = [],
        **kwargs
    ):
        command = get_command()

        logger.set_verbosity(args.verbosity)
        log = logger.get_logger()

        only_test = 'subcommands' in args and \
            'optimize' not in args.subcommands

        flag_config_names = ('modelwrapper_cls', 'dataset_cls', 'compiler_cls',
                             'runtime_cls', 'protocol_cls')
        flag_config_not_none = [getattr(args, name, None) is not None
                                for name in flag_config_names]
        if (args.json_cfg is None and not any(flag_config_not_none)):
            raise argparse.ArgumentError(
                None, "JSON or flag config is required."
            )
        if (args.json_cfg is not None and any(flag_config_not_none)):
            raise argparse.ArgumentError(
                None, "JSON and flag configurations are mutually exclusive. "
                "Please use only one method of configuration.")

        if args.json_cfg is not None:
            return InferenceTester._run_from_json(
                args, only_test, command, log, not_parsed=not_parsed, **kwargs)

        required_args = 3 if 'compiler_cls' in args \
            and 'run_benchmarks_only' not in args else 2
        missing_args = [
            f"'{n}'" for i, n in enumerate(flag_config_names[:required_args])
            if not flag_config_not_none[i]
        ]

        if missing_args:
            raise argparse.ArgumentError(
                None, f"missing required arguments: {', '.join(missing_args)}")

        return InferenceTester._run_from_flags(
            args, only_test, command, log, not_parsed=not_parsed, **kwargs)

    def _run_from_json(
        args: argparse.Namespace,
        only_test: bool,
        command: List[str],
        log,
        not_parsed: List[str] = [],
        **kwargs
    ):
        if not_parsed:
            raise argparse.ArgumentError(
                None,
                f"unrecognized arguments: {' '.join(not_parsed)}"
            )

        with open(args.json_cfg, 'r') as f:
            json_cfg = json.load(f)

        try:
            ret = run_pipeline_json(
                json_cfg,
                args.measurements[0],
                args.verbosity,
                getattr(args, "convert_to_onnx", False),
                command,
                only_test or getattr(args, "run_benchmarks_only", False),
            )
        except ValidationError as ex:
            log.error(f'Validation error: {ex}')
            raise
        except Exception as ex:
            log.error(ex)
            raise

        if not ret:
            return 1
        return ret

    def _run_from_flags(
        args: argparse.Namespace,
        only_test: bool,
        command: List[str],
        log,
        not_parsed: List[str] = [],
        **kwargs
    ):
        modelwrappercls = load_class(args.modelwrapper_cls)
        datasetcls = load_class(args.dataset_cls)
        runtimecls = load_class(args.runtime_cls) \
            if getattr(args, 'runtime_cls', None) else None
        compilercls = load_class(args.compiler_cls) \
            if getattr(args, 'compiler_cls', None) else None
        protocolcls = load_class(args.protocol_cls) \
            if getattr(args, 'protocol_cls', None) else None

        if (compilercls or protocolcls) and not runtimecls:
            raise RuntimeError('Runtime is not provided')

        parser = argparse.ArgumentParser(
            ' '.join(map(lambda x: x.strip(), get_command(with_slash=False))),
            parents=[
                modelwrappercls.form_argparse()[0],
                datasetcls.form_argparse()[0]
            ] + ([runtimecls.form_argparse()[0]] if runtimecls else [])
              + ([compilercls.form_argparse()[0]] if compilercls else [])
              + ([protocolcls.form_argparse()[0]] if protocolcls else [])
        )

        if args.help:
            parser.print_help()
            return 0

        args = parser.parse_args(not_parsed, namespace=args)

        dataset = datasetcls.from_argparse(args)
        model = modelwrappercls.from_argparse(dataset, args)
        compiler = [compilercls.from_argparse(dataset, args)] if compilercls else []  # noqa: E501
        protocol = protocolcls.from_argparse(args) if protocolcls else None
        runtime = runtimecls.from_argparse(
            protocol, args) if runtimecls else None

        try:
            ret = run_pipeline(
                dataset,
                model,
                compiler,
                runtime,
                protocol,
                args.measurements[0],
                args.verbosity,
                getattr(args, "convert_to_onnx", False),
                command,
                only_test or getattr(args, "run_benchmarks_only", False),
            )
        except ValidationError as ex:
            log.error(f'Validation error: {ex}')
            raise
        except Exception as ex:
            log.error(ex)
            raise

        if not ret:
            return 1
        return ret


def main(argv):
    parser, _ = InferenceTester.configure_parser(command=argv[0])
    parser.exit_on_error = False

    args, not_parsed = parser.parse_known_args(argv[1:])

    status = InferenceTester.run(args, not_parsed=not_parsed)
    return status


if __name__ == '__main__':
    main(sys.argv)
