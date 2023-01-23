#!/usr/bin/env python

import argparse
import json
import sys
from pathlib import Path

from pipeline_manager_backend_communication.communication_backend import CommunicationBackend  # noqa: E501
from pipeline_manager_backend_communication.misc_structures import MessageType, Status  # noqa: E501
from kenning.core.measurements import MeasurementsCollector

from kenning.utils.pipeline_manager.misc import get_specification, parse_dataflow, create_dataflow  # noqa: E501
from kenning.utils.pipeline_runner import run_pipeline, parse_json_pipeline


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
    parser.add_argument(
        '--file-path',
        type=Path,
        help='Path where inference output will be stored',
        required=True
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

            if (message_type == MessageType.VALIDATE or
                    message_type == MessageType.RUN or
                    message_type == MessageType.EXPORT):

                dataflow = json.loads(data)
                successful, msg = parse_dataflow(dataflow)

                if not successful:
                    client.send_message(MessageType.ERROR, msg.encode())
                    continue
                try:
                    scenario_tuple = parse_json_pipeline(msg)

                    if message_type == MessageType.EXPORT:
                        with open(args.file_path, 'w') as f:
                            json.dump(msg, f, indent=4)

                    if message_type == MessageType.RUN:
                        MeasurementsCollector.clear()
                        run_pipeline(
                            *scenario_tuple,
                            args.file_path
                        )
                except Exception as ex:
                    client.send_message(MessageType.ERROR, str(ex).encode())
                    continue

                feedback_msg = ''
                if message_type == MessageType.VALIDATE:
                    feedback_msg = 'Successfuly validated'
                elif message_type == MessageType.RUN:
                    feedback_msg = f'Successfuly run. Output saved in {args.file_path}'  # noqa: E501
                elif message_type == MessageType.EXPORT:
                    feedback_msg = f'Successfuly exported. Output saved in {args.file_path}'  # noqa: E501

                client.send_message(
                    MessageType.OK,
                    feedback_msg.encode()
                )
                continue

            if message_type == MessageType.IMPORT:
                pipeline = json.loads(data)
                dataflow = create_dataflow(pipeline)

                client.send_message(
                    MessageType.OK,
                    json.dumps(dataflow).encode(encoding='UTF-8')
                )


if __name__ == '__main__':
    main(sys.argv)
