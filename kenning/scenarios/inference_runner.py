"""
A script that uses the model in inference-only mode with
no performance data collection. It utilizes a DataProvider and
one or more OutputCollectors to supply and gather data to and from the model
and save them or visualize to the user.

It requires implementations of several classes as input:

* DataProvider : provides data for inference
* OutputCollector : interprets and visualizes the output
* Runtime : runs the model with provided data and returns the output
* ModelWrapper : provides methods to convert and interpret input/output data

Each class requires arguments to configure them and provide user settings
"""
import sys
import argparse

from kenning.utils.class_loader import load_class
import kenning.utils.logger as logger


def main(argv):
    parser = argparse.ArgumentParser(argv[0], add_help=False)
    parser.add_argument(
        'modelwrappercls',
        help='ModelWrapper-based class with inference implementation to import',  # noqa: E501
    )
    parser.add_argument(
        'runtimecls',
        help='Runtime-based class with the implementation of model runtime'
    )
    parser.add_argument(
        'dataprovidercls',
        help='DataProvider-based class used for providing data',
    )
    parser.add_argument(
        'outputcollectorcls',
        help='OutputCollector-based class for visualizing and gathering data',
    )
    parser.add_argument(
        '--additional-outputcollector',
        help='Additonal OutputCollector-based class (can be called multiple times)',  # noqa: E501
        action='append'
    )
    parser.add_argument(
        '--verbosity',
        help='Verbosity level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
    )

    args, _ = parser.parse_known_args(argv[1:])

    log = logger.set_verbosity(args.verbosity)
    log = logger.get_logger()

    modelwrappercls = load_class(args.modelwrappercls)
    runtimecls = load_class(args.runtimecls)
    dataprovidercls = load_class(args.dataprovidercls)
    outputcollectorcls = [load_class(args.outputcollectorcls)]

    if args.additional_outputcollector:
        for i in args.additional_outputcollector:
            outputcollectorcls.append(load_class(i))

    parser = argparse.ArgumentParser(
        argv[0],
        parents=[
            parser,
            modelwrappercls.form_argparse()[0],
            runtimecls.form_argparse()[0],
            dataprovidercls.form_argparse()[0]
        ] + ([i.form_argparse()[0] for i in outputcollectorcls])
    )

    args = parser.parse_args(argv[1:])

    dataprovider = dataprovidercls.from_argparse(args)
    model = modelwrappercls.from_argparse(dataprovider, args)
    runtime = runtimecls.from_argparse(None, args)
    outputcollectors = [
        o.from_argparse(args) for o in outputcollectorcls
    ]

    runtime.prepare_model(None)
    log.info("Starting inference session")
    try:
        while True:
            log.debug("Fetching data")
            unconverted_inp = dataprovider.get_input()
            preprocessed_input = dataprovider.preprocess_input(unconverted_inp)

            log.debug("Converting to bytes and setting up model input")
            inp = model.convert_input_to_bytes(
                model.preprocess_input(preprocessed_input)
            )
            runtime.prepare_input(inp)

            log.debug("Running inference")
            runtime.run()

            log.debug("Converting output from bytes")
            res = model.postprocess_outputs(
                model.convert_output_from_bytes(runtime.upload_output(inp))
            )

            log.debug("Sending data to collectors")
            for i in outputcollectors:
                i.return_output(unconverted_inp, res)
    except KeyboardInterrupt:
        log.info("Interrupt signal caught, shutting down (press CTRL-C again to force quit)")  # noqa E501
        dataprovider.detach_from_source()
        for o in outputcollectors:
            o.detach_from_output()


if __name__ == '__main__':
    main(sys.argv)
