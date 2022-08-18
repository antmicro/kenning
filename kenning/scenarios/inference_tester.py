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

import sys
import argparse
import tempfile
from pathlib import Path

from kenning.utils.class_loader import load_class, get_command
import kenning.utils.logger as logger
from kenning.core.measurements import MeasurementsCollector
from kenning.utils.args_manager import serialize_inference


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

    args = parser.parse_args(argv[1:])

    logger.set_verbosity(args.verbosity)
    logger.get_logger()

    dataset = datasetcls.from_argparse(args)
    model = modelwrappercls.from_argparse(dataset, args)
    compiler = compilercls.from_argparse(dataset, args) if compilercls else None  # noqa: E501
    protocol = protocolcls.from_argparse(args) if protocolcls else None
    runtime = runtimecls.from_argparse(protocol, args) if runtimecls else None

    modelpath = model.get_path()
    inputspec, inputdtype = model.get_input_spec()

    modelframeworktuple = model.get_framework_and_version()

    if compiler:
        compilerframeworktuple = compiler.get_framework_and_version()
        compiler_info = [
            {
                'compiler_framework': compilerframeworktuple[0],
                'compiler_version': compilerframeworktuple[1]
            }
        ]
    else:
        compiler_info = []

    MeasurementsCollector.measurements += {
        'model_framework': modelframeworktuple[0],
        'model_version': modelframeworktuple[1],
        'compilers': compiler_info,
        'command': command,
        'build_cfg': serialize_inference(
            dataset,
            model,
            compiler,
            protocol,
            runtime
        )
    }

    # TODO add method for providing metadata to dataset
    if hasattr(dataset, 'classnames'):
        MeasurementsCollector.measurements += {
            'class_names': [val for val in dataset.get_class_names()]
        }

    if args.convert_to_onnx:
        modelpath = args.convert_to_onnx
        model.save_to_onnx(modelpath)

    if compiler:
        # TODO make use of --model-framework parameter or make it optional and
        # use it only if specified
        if args.convert_to_onnx:
            format = 'onnx'
        else:
            format = compiler.consult_model_type(model)
            if format == 'onnx':
                modelpath = Path(tempfile.NamedTemporaryFile().name)
                model.save_to_onnx(modelpath)

        compiler.set_input_type(format)
        compiler.compile(modelpath, inputspec, inputdtype)
        modelpath = compiler.compiled_model_path

    if runtime:
        if protocol:
            ret = runtime.run_client(dataset, model, modelpath)
        else:
            ret = runtime.run_locally(dataset, model, modelpath)
    else:
        model.test_inference()
        ret = True

    if not ret:
        return 1

    MeasurementsCollector.save_measurements(args.output)
    return 0


if __name__ == '__main__':
    main(sys.argv)
