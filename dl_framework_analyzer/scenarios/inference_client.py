import sys
import argparse
from pathlib import Path

from dl_framework_analyzer.utils.class_loader import load_class
import dl_framework_analyzer.utils.logger as logger


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


if __name__ == '__main__':
    main(sys.argv)
