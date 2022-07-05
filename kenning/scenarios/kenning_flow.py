import argparse
import json
import sys
from kenning.core.flow import KenningFlow
from kenning.utils.class_loader import get_command


def main(argv):
    _ = get_command(argv)

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

    with open(args.jsoncfg, 'r') as f:
        json_cfg = json.load(f)

    flow: KenningFlow = KenningFlow.from_json(json_cfg)
    _ = flow.process()

    print('done')
    return 3


if __name__ == '__main__':
    main(sys.argv)
