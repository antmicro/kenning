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

from dl_framework_analyzer.utils.class_loader import load_class
import dl_framework_analyzer.utils.logger as logger
from dl_framework_analyzer.core.measurements import MeasurementsCollector
from dl_framework_analyzer.core.drawing import create_line_plot
from dl_framework_analyzer.core.drawing import draw_confusion_matrix


def main(argv):
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
        'protocolcls',
        help='RuntimeProtocol-based class with the implementation of communication between inference tester and inference runner',  # noqa: E501
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
        'onnxmodelpath',
        help='Path to the intermediate ONNX model',
        type=Path
    )
    parser.add_argument(
        'output',
        help='The path to the output directory',
        type=Path
    )
    parser.add_argument(
        'reportname',
        help='The name of the generated report, used as prefix for images',
        type=str
    )
    parser.add_argument(
        '--resources-dir',
        help='The path to the directory with resources',
        type=Path
    )
    parser.add_argument(
        '--verbosity',
        help='Verbosity level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
    )

    args, _ = parser.parse_known_args(argv[1:])

    modelwrappercls = load_class(args.modelwrappercls)
    modelcompilercls = load_class(args.modelcompilercls)
    protocolcls = load_class(args.protocolcls)
    runtimecls = load_class(args.runtimecls)
    datasetcls = load_class(args.datasetcls)

    parser = argparse.ArgumentParser(
        argv[0],
        parents=[
            parser,
            modelwrappercls.form_argparse()[0],
            modelcompilercls.form_argparse()[0],
            protocolcls.form_argparse()[0],
            runtimecls.form_argparse()[0],
            datasetcls.form_argparse()[0]
        ]
    )

    args = parser.parse_args(argv[1:])

    if args.resources_dir is None:
        args.resources_dir = Path(args.output / 'img')

    args.output.mkdir(parents=True, exist_ok=True)
    args.resources_dir.mkdir(parents=True, exist_ok=True)

    logger.set_verbosity(args.verbosity)
    logger.get_logger()

    dataset = datasetcls.from_argparse(args)
    model = modelwrappercls.from_argparse(dataset, args)
    compiler = modelcompilercls.from_argparse(args)
    protocol = protocolcls.from_argparse(args)
    runtime = runtimecls.from_argparse(protocol, args)

    model.save_to_onnx(args.onnxmodelpath)

    inputspec, inputdtype = model.get_input_spec()

    compiler.compile(args.onnxmodelpath, inputspec, inputdtype)
    runtime.run_client(dataset, model, compiler.compiled_model_path)

    reportname = args.reportname

    measurementsdata = MeasurementsCollector.measurements.data

    batchtime = args.resources_dir / f'{reportname}-batchtime.png'
    memusage = args.resources_dir / f'{reportname}-memoryusage.png'
    gpumemusage = args.resources_dir / f'{reportname}-gpumemoryusage.png'
    gpuusage = args.resources_dir / f'{reportname}-gpuutilization.png'
    confusionmatrix = args.resources_dir / f'{reportname}-conf-matrix.png'
    create_line_plot(
        batchtime,
        'Target inference time for batches',
        'Time', 's',
        'Inference time', 's',
        measurementsdata['target_inference_step_timestamp'],
        measurementsdata['target_inference_step'])
    create_line_plot(
        memusage,
        'Memory usage over benchmark',
        'Time', 's',
        'Memory usage', '%',
        measurementsdata['session_utilization_timestamp'],
        measurementsdata['session_utilization_mem_percent'])
    create_line_plot(
        gpumemusage,
        'GPU Memory usage over benchmark',
        'Time', 's',
        'Memory usage', '%',
        measurementsdata['session_utilization_gpu_timestamp'],
        measurementsdata['session_utilization_gpu_mem_utilization'])
    create_line_plot(
        gpuusage,
        'GPU usage over benchmark',
        'Time', 's',
        'Memory usage', '%',
        measurementsdata['session_utilization_gpu_timestamp'],
        measurementsdata['session_utilization_gpu_utilization'])
    if 'eval_confusion_matrix' in measurementsdata:
        draw_confusion_matrix(
            measurementsdata['eval_confusion_matrix'],
            confusionmatrix,
            'Confusion matrix',
            [dataset.classnames[i] for i in range(dataset.numclasses)],
            True
        )

    MeasurementsCollector.measurements.data['eval_confusion_matrix'] = MeasurementsCollector.measurements.data['eval_confusion_matrix'].tolist()  # noqa: E501
    with open(args.output / 'measurements.json', 'w') as measurementsfile:
        json.dump(
            MeasurementsCollector.measurements.data,
            measurementsfile,
            indent=2
        )


if __name__ == '__main__':
    main(sys.argv)
