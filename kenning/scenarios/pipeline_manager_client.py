#!/usr/bin/env python

import json
import argparse
import sys

from pipeline_manager_backend_communication.communication_backend import CommunicationBackend   # noqa: E501
from pipeline_manager_backend_communication.misc_structures import MessageType, Status  # noqa: E501
from kenning.utils.pipeline_manager.misc import get_specification


def main(argv):
    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument(
        '--host',
        type=str,
        help='The address of the Pipeline Manager Server',
        default='127.0.0.1'
    )
    parser.add_argument(
        '--port',
        type=int,
        help='The port of the Pipeline Manager Server',
        default=9000
    )
    args, _ = parser.parse_known_args(argv[1:])

    client = CommunicationBackend(args.host, args.port)
    client.initialize_client()

    while client.client_socket:
        status, message = client.wait_for_message()
        if status == Status.DATA_READY:
            message_type, data = message

            if message_type == MessageType.SPECIFICATION:
                client.send_message(
                    MessageType.OK,
                    json.dumps(get_specification()).encode(encoding='UTF-8')
                )


if __name__ == '__main__':
    main(sys.argv)
