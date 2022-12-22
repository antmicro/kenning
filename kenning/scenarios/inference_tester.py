#!/usr/bin/env python

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
compilation and benchmark process
"""

import argparse
import sys
from pathlib import Path

from jsonschema.exceptions import ValidationError

from kenning.utils.class_loader import get_command, load_class
from kenning.utils.pipeline_runner import run_pipeline
import kenning.utils.logger as logger


def main(argv):
    command = get_command(argv)

    parser = argparse.ArgumentParser(argv[0], add_help=False)
    parser.add_argument(
        'modelwrappercls',
        help='ModelWrapper-based class with inference implementation to import',  # noqa: E501
    )
    parser.add_argument(
        'datasetcls',
        help='Dataset-based class with dataset to import',
    )
    parser.add_argument(
        '--runtime-cls',
        help='Runtime-based class with the implementation of model runtime'
    )
    parser.add_argument(
        '--compiler-cls',
        help='Optimizer-based class with compiling routines to import'
    )
    parser.add_argument(
        '--protocol-cls',
        help='RuntimeProtocol-based class with the implementation of communication between inference tester and inference runner',  # noqa: E501
    )

    args, _ = parser.parse_known_args(argv[1:])

    modelwrappercls = load_class(args.modelwrappercls)
    datasetcls = load_class(args.datasetcls)
    runtimecls = load_class(args.runtime_cls) if args.runtime_cls else None
    compilercls = load_class(args.compiler_cls) if args.compiler_cls else None  # noqa: E501
    protocolcls = load_class(args.protocol_cls) if args.protocol_cls else None

    if (compilercls or protocolcls) and not runtimecls:
        raise RuntimeError('Runtime is not provided')

    parser = argparse.ArgumentParser(
        argv[0],
        parents=[
            parser,
            modelwrappercls.form_argparse()[0],
            datasetcls.form_argparse()[0]
        ] + ([runtimecls.form_argparse()[0]] if runtimecls else [])
          + ([compilercls.form_argparse()[0]] if compilercls else [])
          + ([protocolcls.form_argparse()[0]] if protocolcls else [])
    )

    parser.add_argument(
        'output',
        help='The path to the output JSON file with measurements',
        type=Path
    )
    parser.add_argument(
        '--convert-to-onnx',
        help='Before compiling the model, convert it to ONNX and use in compilation (provide a path to save here)',  # noqa: E501
        type=Path
    )
    parser.add_argument(
        '--verbosity',
        help='Verbosity level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
    )
    parser.add_argument(
        '--run-benchmarks-only',
        help='Instead of running the full compilation and testing flow, only testing of the model is executed',  # noqa: E501
        action='store_true'
    )

    args = parser.parse_args(argv[1:])

    logger.set_verbosity(args.verbosity)
    log = logger.get_logger()

    dataset = datasetcls.from_argparse(args)
    model = modelwrappercls.from_argparse(dataset, args)
    compiler = compilercls.from_argparse(dataset, args) if compilercls else None  # noqa: E501
    protocol = protocolcls.from_argparse(args) if protocolcls else None
    runtime = runtimecls.from_argparse(protocol, args) if runtimecls else None

    try:
        ret = run_pipeline(
            dataset,
            model,
            [compiler],
            runtime,
            protocol,
            args.output,
            args.verbosity,
            args.convert_to_onnx,
            command,
            args.run_benchmarks_only
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


if __name__ == '__main__':
    main(sys.argv)
