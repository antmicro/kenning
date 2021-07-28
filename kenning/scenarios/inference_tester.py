#!/usr/bin/env python

"""
A script that runs inference client.

It requires implementations of several classes as input:

* ModelWrapper - wraps the model that will be compiled and executed on hardware
* ModelCompiler - wraps the compiling routines for the deep learning model
* RuntimeProtocol - describes the protocol over which the communication is
  performed
* Dataset - provides data for benchmarking

Each of those classes require specific set or arguments to configure the
compilation and benchmark process.
"""

import sys
import argparse
from pathlib import Path
import json

from kenning.utils.class_loader import load_class, get_command
import kenning.utils.logger as logger
from kenning.core.measurements import MeasurementsCollector


def main(argv):
    command = get_command(argv)

    parser = argparse.ArgumentParser(argv[0], add_help=False)
    parser.add_argument(
        'modelwrappercls',
        help='ModelWrapper-based class with inference implementation to import',  # noqa: E501
    )
    parser.add_argument(
        'modelcompilercls',
        help='ModelCompiler-based class with compiling routines to import'
    )
    parser.add_argument(
        'runtimecls',
        help='Runtime-based class with the implementation of model runtime'
    )
    parser.add_argument(
        'datasetcls',
        help='Dataset-based class with dataset to import',
    )
    parser.add_argument(
        '--protocol-cls',
        help='RuntimeProtocol-based class with the implementation of communication between inference tester and inference runner',  # noqa: E501
    )

    args, _ = parser.parse_known_args(argv[1:])

    modelwrappercls = load_class(args.modelwrappercls)
    modelcompilercls = load_class(args.modelcompilercls)
    runtimecls = load_class(args.runtimecls)
    datasetcls = load_class(args.datasetcls)
    if args.protocol_cls:
        protocolcls = load_class(args.protocol_cls)
    else:
        protocolcls = None

    parser = argparse.ArgumentParser(
        argv[0],
        parents=[
            parser,
            modelwrappercls.form_argparse()[0],
            modelcompilercls.form_argparse()[0],
            runtimecls.form_argparse()[0],
            datasetcls.form_argparse()[0]
        ] + ([protocolcls.form_argparse()[0]] if protocolcls else [])
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

    args = parser.parse_args(argv[1:])

    logger.set_verbosity(args.verbosity)
    logger.get_logger()

    dataset = datasetcls.from_argparse(args)
    model = modelwrappercls.from_argparse(dataset, args)
    compiler = modelcompilercls.from_argparse(dataset, args)
    protocol = protocolcls.from_argparse(args) if protocolcls else None
    runtime = runtimecls.from_argparse(protocol, args)

    modelpath = model.get_path()
    if args.convert_to_onnx:
        model.save_to_onnx(args.convert_to_onnx)
        modelpath = args.convert_to_onnx

    inputspec, inputdtype = model.get_input_spec()

    modelframeworktuple = model.get_framework_and_version()
    compilerframeworktuple = compiler.get_framework_and_version()

    MeasurementsCollector.measurements += {
        'model_framework': modelframeworktuple[0],
        'model_version': modelframeworktuple[1],
        'compiler_framework': compilerframeworktuple[0],
        'compiler_version': compilerframeworktuple[1],
        'command': command
    }

    # TODO add method for providing metadata to dataset
    if hasattr(dataset, 'classnames'):
        MeasurementsCollector.measurements += {
            'class_names': [val for val in dataset.get_class_names()]
        }

    compiler.compile(modelpath, inputspec, inputdtype)

    if protocol:
        ret = runtime.run_client(dataset, model, compiler.compiled_model_path)
    else:
        ret = runtime.run_locally(dataset, model, compiler.compiled_model_path)

    if not ret:
        return 1

    if 'eval_confusion_matrix' in MeasurementsCollector.measurements.data:
        MeasurementsCollector.measurements.data['eval_confusion_matrix'] = MeasurementsCollector.measurements.data['eval_confusion_matrix'].tolist()  # noqa: E501
    with open(args.output, 'w') as measurementsfile:
        json.dump(
            MeasurementsCollector.measurements.data,
            measurementsfile,
            indent=2
        )
    return 0


if __name__ == '__main__':
    main(sys.argv)
