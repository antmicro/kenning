#!/usr/bin/env python

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A script that runs inference server based on a json file.

It requires implementations of several classes as input:

* RuntimeProtocol - provides routines for communicating with the client
* Runtime - provides implementation of inference runtime

Each of those classes require specific set or arguments to configure the
compilation and benchmark process.
"""

import sys
import argparse
import signal
import json

from kenning.utils.class_loader import load_class
import kenning.utils.logger as logger


def main(argv):
    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument(
        'jsoncfg',
        help='The path to the input JSON file with configuration'
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

    protocolcfg = json_cfg['runtime_protocol']
    runtimecfg = json_cfg['runtime']

    protocolcls = load_class(protocolcfg['type'])
    runtimecls = load_class(runtimecfg['type'])

    protocol = protocolcls.from_json(protocolcfg['parameters'])
    runtime = runtimecls.from_json(protocol, runtimecfg['parameters'])

    logger.set_verbosity(args.verbosity)
    logger.get_logger()

    formersighandler = signal.getsignal(signal.SIGINT)

    def sigint_handler(sig, frame):
        runtime.close_server()
        runtime.protocol.log.info('Closing application (press Ctrl-C again for force closing)...')  # noqa: E501
        signal.signal(signal.SIGINT, formersighandler)

    signal.signal(signal.SIGINT, sigint_handler)

    runtime.run_server()


if __name__ == '__main__':
    main(sys.argv)
