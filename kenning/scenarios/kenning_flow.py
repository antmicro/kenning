# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
from kenning.core.flow import KenningFlow
from kenning.utils import logger


def main(argv):
    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument(
        'jsoncfg',
        help='The path to the input JSON file with configuration of the graph'
    )
    parser.add_argument(
        '--verbosity',
        help='Verbosity level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
    )

    args, _ = parser.parse_known_args(argv[1:])

    logger.set_verbosity(args.verbosity)
    log = logger.get_logger()

    with open(args.jsoncfg, 'r') as f:
        json_cfg = json.load(f)

    flow: KenningFlow = KenningFlow.from_json(json_cfg)
    _ = flow.process()

    log.info('Processing has finished')
    return 0


if __name__ == '__main__':
    main(sys.argv)
