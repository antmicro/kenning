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
import json
import sys
from pathlib import Path

from kenning.utils.class_loader import get_command
from kenning.utils.pipeline_runner import run_pipeline_json


def main(argv):
    command = get_command(argv)

    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument(
        'jsoncfg',
        help='The path to the input JSON file with configuration of the inference'  # noqa: E501
    )
    parser.add_argument(
        'output',
        help='The path to the output JSON file with measurements',
        type=Path,
    )
    parser.add_argument(
        '--verbosity',
        help='Verbosity level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
    )
    parser.add_argument(
        '--convert-to-onnx',
        help='Before compiling the model, convert it to ONNX and use in the inference (provide a path to save here)',  # noqa: E501
        type=Path
    )
    parser.add_argument(
        '--run-benchmarks-only',
        help='Instead of running the full compilation and testing flow, only testing of the model is executed',  # noqa: E501
        action='store_true'
    )

    args, _ = parser.parse_known_args(argv[1:])

    with open(args.jsoncfg, 'r') as f:
        json_cfg = json.load(f)

    return run_pipeline_json(
        json_cfg,
        args.output,
        args.verbosity,
        args.convert_to_onnx,
        command,
        args.run_benchmarks_only
    )


if __name__ == '__main__':
    main(sys.argv)
