# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

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

Each class requires arguments to configure them and provide user settings.
"""
import argparse
import sys
from typing import List

from kenning.core.outputcollector import OutputCollector
from kenning.utils.class_loader import load_class
from kenning.utils.logger import KLogger


def check_closing_conditions(outputcollectors: List[OutputCollector]) -> bool:
    """
    Checks closing conditions of outputcollectors.

    Parameters
    ----------
    outputcollectors : List[OutputCollector]
        List of outputcollectors to check.

    Returns
    -------
    bool
        True if any OutputCollector.should_close() method returned True.
    """
    if any(i.should_close() for i in outputcollectors):
        return True
    return False


def main(argv):  # noqa: D103
    parser = argparse.ArgumentParser(argv[0], add_help=False)
    parser.add_argument(
        "modelwrappercls",
        help="ModelWrapper-based class with inference implementation to import",  # noqa: E501
    )
    parser.add_argument(
        "runtimecls",
        help="Runtime-based class with the implementation of model runtime",
    )
    parser.add_argument(
        "dataprovidercls",
        help="DataProvider-based class used for providing data",
    )
    parser.add_argument(
        "--output-collectors",
        help="List to the OutputCollector-based classes where the results will be passed",  # noqa: E501
        required=True,
        nargs="+",
    )

    parser.add_argument(
        "--verbosity",
        help="Verbosity level",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )

    args, _ = parser.parse_known_args(argv[1:])

    KLogger.set_verbosity(args.verbosity)

    modelwrappercls = load_class(args.modelwrappercls)
    runtimecls = load_class(args.runtimecls)
    dataprovidercls = load_class(args.dataprovidercls)
    outputcollectorcls = [load_class(i) for i in args.output_collectors]

    parser = argparse.ArgumentParser(
        argv[0],
        parents=[
            parser,
            modelwrappercls.form_argparse(args)[0],
            runtimecls.form_argparse(args)[0],
            dataprovidercls.form_argparse(args)[0],
        ]
        + ([i.form_argparse(args)[0] for i in outputcollectorcls]),
    )

    args = parser.parse_args(argv[1:])

    dataprovider = dataprovidercls.from_argparse(args)
    model = modelwrappercls.from_argparse(None, args)
    runtime = runtimecls.from_argparse(None, args)
    outputcollectors = [o.from_argparse(args) for o in outputcollectorcls]

    runtime.prepare_model(None)
    KLogger.info("Starting inference session")
    try:
        # check against the output collectors
        # if an exit condition was reached in any of them
        while not check_closing_conditions(outputcollectors):
            KLogger.debug("Fetching data")
            unconverted_inp = dataprovider.fetch_input()
            preprocessed_input = dataprovider.preprocess_input(unconverted_inp)

            KLogger.debug("Setting up model input")
            runtime.load_input([model.preprocess_input(preprocessed_input)])

            KLogger.debug("Running inference")
            runtime.run()

            KLogger.debug("Postprocessing output")
            res = model.postprocess_outputs(runtime.extract_output())

            KLogger.debug("Sending data to collectors")
            for i in outputcollectors:
                i.process_output(unconverted_inp, res)
    except KeyboardInterrupt:
        KLogger.info(
            "Interrupt signal caught, shutting down (press CTRL-C again to "
            "force quit)"
        )
        dataprovider.detach_from_source()
        for o in outputcollectors:
            o.detach_from_output()


if __name__ == "__main__":
    main(sys.argv)
