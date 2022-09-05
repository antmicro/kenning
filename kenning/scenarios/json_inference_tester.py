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
import json
import tempfile
from pathlib import Path

from kenning.utils.class_loader import load_class, get_command
import kenning.utils.logger as logger
from kenning.core.measurements import MeasurementsCollector


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

    args, _ = parser.parse_known_args(argv[1:])

    with open(args.jsoncfg, 'r') as f:
        json_cfg = json.load(f)

    modelwrappercfg = json_cfg['model_wrapper']
    datasetcfg = json_cfg['dataset']
    runtimecfg = (
        json_cfg['runtime']
        if 'runtime' in json_cfg else None
    )
    optimizerscfg = (
        json_cfg['optimizers']
        if 'optimizers' in json_cfg else []
    )
    protocolcfg = (
        json_cfg['runtime_protocol']
        if 'runtime_protocol' in json_cfg else None
    )

    if (optimizerscfg or protocolcfg) and not runtimecfg:
        raise RuntimeError('Runtime is not provided')

    modelwrappercls = load_class(modelwrappercfg['type'])
    datasetcls = load_class(datasetcfg['type'])
    runtimecls = (
        load_class(runtimecfg['type'])
        if runtimecfg else None
    )
    optimizerscls = [load_class(cfg['type']) for cfg in optimizerscfg]
    protocolcls = (
        load_class(protocolcfg['type'])
        if protocolcfg else None
    )

    logger.set_verbosity(args.verbosity)
    logger.get_logger()

    dataset = datasetcls.from_json(datasetcfg['parameters'])
    model = modelwrappercls.from_json(dataset, modelwrappercfg['parameters'])
    optimizers = [
        cls.from_json(dataset, cfg['parameters'])
        for cfg, cls in zip(optimizerscfg, optimizerscls)
    ]
    protocol = (
        protocolcls.from_json(protocolcfg['parameters'])
        if protocolcls else None
    )
    runtime = (
        runtimecls.from_json(protocol, runtimecfg['parameters'])
        if runtimecls else None
    )

    logger.set_verbosity(args.verbosity)
    logger.get_logger()

    modelframeworktuple = model.get_framework_and_version()

    MeasurementsCollector.measurements += {
        'model_framework': modelframeworktuple[0],
        'model_version': modelframeworktuple[1],
        'compilers': [
            {
                'compiler_framework': optimizer.get_framework_and_version()[0],
                'compiler_version': optimizer.get_framework_and_version()[1]
            }
            for optimizer in optimizers
        ],
        'command': command,
        'build_cfg': json_cfg
    }

    # TODO add method for providing metadata to dataset
    if hasattr(dataset, 'classnames'):
        MeasurementsCollector.measurements += {
            'class_names': [val for val in dataset.get_class_names()]
        }

    modelpath = model.get_path()

    prev_block = model
    if args.convert_to_onnx:
        modelpath = args.convert_to_onnx
        prev_block.save_to_onnx(modelpath)

    for i in range(len(optimizers)):
        next_block = optimizers[i]

        format = next_block.consult_model_type(
            prev_block,
            force_onnx=(args.convert_to_onnx and prev_block == model)
        )

        if (format == 'onnx' and prev_block == model) and \
                not args.convert_to_onnx:
            modelpath = Path(tempfile.NamedTemporaryFile().name)
            prev_block.save_to_onnx(modelpath)

        prev_block.save_io_specification(modelpath)
        next_block.set_input_type(format)
        next_block.compile(modelpath)

        prev_block = next_block
        modelpath = prev_block.compiled_model_path

    if not optimizers:
        model.save_io_specification(modelpath)

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
