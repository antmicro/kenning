#!/usr/bin/env python

"""
A script that runs inference server.

It requires implementations of several classes as input:

* RuntimeProtocol - provides routines for communicating with the client
* Runtime - provides implementation of inference runtime

Each of those classes require specific set or arguments to configure the
compilation and benchmark process.
"""

import sys
import argparse
import signal

from edge_ai_tester.utils.class_loader import load_class
import edge_ai_tester.utils.logger as logger


def main(argv):
    parser = argparse.ArgumentParser(argv[0], add_help=False)
    parser.add_argument(
        'protocolcls',
        help='RuntimeProtocol-based class with the implementation of communication between inference tester and inference runner',  # noqa: E501
    )
    parser.add_argument(
        'runtimecls',
        help='Runtime-based class with the implementation of model runtime'
    )
    parser.add_argument(
        '--verbosity',
        help='Verbosity level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
    )

    args, _ = parser.parse_known_args(argv[1:])

    protocolcls = load_class(args.protocolcls)
    runtimecls = load_class(args.runtimecls)

    parser = argparse.ArgumentParser(
        argv[0],
        parents=[
            parser,
            protocolcls.form_argparse()[0],
            runtimecls.form_argparse()[0],
        ]
    )

    args = parser.parse_args(argv[1:])

    logger.set_verbosity(args.verbosity)
    logger.get_logger()

    protocol = protocolcls.from_argparse(args)
    runtime = runtimecls.from_argparse(protocol, args)

    formersighandler = signal.getsignal(signal.SIGINT)

    def sigint_handler(sig, frame):
        runtime.close_server()
        runtime.protocol.log.info('Closing application (press Ctrl-C again for force closing)...')  # noqa: E501
        signal.signal(signal.SIGINT, formersighandler)

    signal.signal(signal.SIGINT, sigint_handler)

    runtime.run_server()


if __name__ == '__main__':
    main(sys.argv)
