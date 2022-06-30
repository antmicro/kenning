#!/usr/bin/env python

"""
A script that runs inference client based on a json file.

It requires implementations of several classes as input:

* ModelWrapper - wraps the model that will be compiled and executed on hardware
* Optimizer - wraps the compiling routines for the deep learning model
* RuntimeProtocol - describes the protocol over which the communication is
  performed
* Dataset - provides data for benchmarking

Each of those classes require specific set or arguments to configure the
compilation and benchmark process.
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

    args, _ = parser.parse_known_args(argv[1:])

    with open(args.jsoncfg, 'r') as f:
        json_cfg = json.load(f)

    datasetcfg = json_cfg['dataset']
    modelwrappercfg = json_cfg['model_wrapper']
    optimizerscfg = json_cfg['optimizers']
    runtimecfg = json_cfg['runtime']
    protocolcfg = (
        json_cfg['runtime_protocol']
        if 'runtime_protocol' in json_cfg else None
    )

    datasetcls = load_class(datasetcfg['type'])
    modelwrappercls = load_class(modelwrappercfg['type'])
    optimizerscls = [load_class(cfg['type']) for cfg in optimizerscfg]
    runtimecls = load_class(runtimecfg['type'])
    protocolcls = (
        load_class(protocolcfg['type'])
        if protocolcfg else None
    )

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
    runtime = runtimecls.from_json(protocol, runtimecfg['parameters'])

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
    inputspec, inputdtype = model.get_input_spec()

    prev_block = model
    for i in range(len(optimizers)):
        next_block = optimizers[i]

        format = next_block.consult_model_type(prev_block)
        if format == 'onnx' and prev_block == model:
            modelpath = tempfile.NamedTemporaryFile().name
            prev_block.save_to_onnx(modelpath)

        next_block.set_input_type(format)
        next_block.compile(modelpath, inputspec, inputdtype)

        prev_block = next_block
        modelpath = prev_block.compiled_model_path
        inputdtype = prev_block.get_inputdtype()

    compiler = next_block
    if protocol:
        ret = runtime.run_client(dataset, model, compiler.compiled_model_path)
    else:
        ret = runtime.run_locally(dataset, model, compiler.compiled_model_path)
    if not ret:
        return 1

    MeasurementsCollector.save_measurements(args.output)
    return 0


if __name__ == '__main__':
    main(sys.argv)
