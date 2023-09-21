#!/usr/bin/env python

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
A script for connecting Kenning with Pipeline Manager server.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple, Dict, Optional, List

from kenning.cli.command_template import (
    ArgumentsGroups, CommandTemplate, GROUP_SCHEMA, VISUAL_EDITOR)
from kenning.core.measurements import MeasurementsCollector
from kenning.pipeline_manager.core import BaseDataflowHandler
from kenning.pipeline_manager.flow_handler import KenningFlowHandler
from kenning.pipeline_manager.pipeline_handler import PipelineHandler
import kenning.utils.logger as logger
from kenning.utils.excepthook import find_missing_optional_dependency, \
    MissingKenningDependencies
from kenning.utils.logger import Callback, TqdmCallback
from jsonschema.exceptions import ValidationError


class PipelineManagerClient(CommandTemplate):
    parse_all = True
    description = __doc__.split('\n\n')[0]

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Optional[ArgumentsGroups] = None,
    ) -> Tuple[argparse.ArgumentParser, ArgumentsGroups]:
        parser, groups = super(
            PipelineManagerClient,
            PipelineManagerClient
        ).configure_parser(parser, command, types, groups)

        ve_group = parser.add_argument_group(
            GROUP_SCHEMA.format(VISUAL_EDITOR))

        ve_group.add_argument(
            '--host',
            type=str,
            help='The address of the Pipeline Manager Server',
            default='127.0.0.1',
        )
        ve_group.add_argument(
            '--port',
            type=int,
            help='The port of the Pipeline Manager Server',
            default=9000,
        )
        ve_group.add_argument(
            '--file-path',
            type=Path,
            help='Path where results of model benchmarking will be stored (pipeline mode only)',  # noqa: E501
            required=True
        )
        ve_group.add_argument(
            '--workspace-dir',
            type=Path,
            help='Directory where the frontend sources should be stored',
            required=True
        )
        ve_group.add_argument(
            '--spec-type',
            type=str,
            help='Type of graph that should be represented in a Pipeline Manager - can choose between optimization pipeline or Kenningflow',  # noqa: E501
            choices=('pipeline', 'flow'),
            default='pipeline',
        )
        ve_group.add_argument(
            '--layout',
            help='Autolayout algorithm to use in Pipeline Manager',
            type=str,
            default='CytoscapeEngine - breadthfirst'
        )

        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, **kwargs):
        from pipeline_manager_backend_communication.communication_backend import CommunicationBackend  # noqa: E501
        from pipeline_manager_backend_communication.misc_structures import MessageType, Status  # noqa: E501
        from pipeline_manager.backend.run_in_parallel import \
            start_server_in_parallel, stop_parallel_server
        from pipeline_manager import frontend_builder

        def parse_message(
            dataflow_handler: BaseDataflowHandler,
            message_type: MessageType,
            data: bytes,
            output_file_path: Path,
        ) -> Tuple[MessageType, bytes]:
            """
            Uses dataflow_handler to parse the incoming data from Pipeline
            Manager according to the action that is to be performed.

            Parameters
            ----------
            dataflow_handler : BaseDataflowHandler
                Used to convert to and from Pipeline Manager JSON formats,
                create and run dataflows defined in manager.
            message_type : MessageType
                Action requested by the Pipeline Manager to perform.
            data : bytes
                Data send by Manager.
            output_file_path : Path
                Path where the optional output will be saved.

            Returns
            -------
            Tuple[MessageType, bytes]
                Return answer to send to the Manager.
            """
            from pipeline_manager_backend_communication.misc_structures import MessageType  # noqa: E501

            if message_type == MessageType.SPECIFICATION:
                specification = dataflow_handler.get_specification()
                feedback_msg = json.dumps(specification)

            elif (
                message_type == MessageType.VALIDATE
                or message_type == MessageType.RUN
                or message_type == MessageType.EXPORT
            ):
                dataflow = json.loads(data)
                successful, msg = dataflow_handler.parse_dataflow(dataflow)

                if not successful:
                    return MessageType.ERROR, msg.encode()
                try:
                    prepared_runner = dataflow_handler.parse_json(msg)

                    if message_type == MessageType.RUN:
                        MeasurementsCollector.clear()
                        dataflow_handler.run_dataflow(
                            prepared_runner, output_file_path
                        )
                    else:
                        if message_type == MessageType.EXPORT:
                            with open(output_file_path, 'w') as f:
                                json.dump(msg, f, indent=4)

                        # runner is created without processing it through
                        # 'run_dataflow', it should be destroyed manually.
                        dataflow_handler.destroy_dataflow(prepared_runner)
                except Exception as ex:
                    return MessageType.ERROR, str(ex).encode()

                if message_type == MessageType.VALIDATE:
                    feedback_msg = 'Successfully validated'
                elif message_type == MessageType.RUN:
                    feedback_msg = f'Successfully run. Output saved in {output_file_path}'  # noqa: E501
                elif message_type == MessageType.EXPORT:
                    feedback_msg = f'Successfully exported. Output saved in {output_file_path}'  # noqa: E501

            elif message_type == MessageType.IMPORT:
                pipeline = json.loads(data)
                dataflow = dataflow_handler.create_dataflow(pipeline)
                feedback_msg = json.dumps(dataflow)

            return MessageType.OK, feedback_msg.encode(encoding='UTF-8')

        def send_progress(state: Dict, client: CommunicationBackend):
            """
            Sends progress message to Pipeline Manager.

            Parameters
            ----------
            state : Dict
                The `format_dict` that comes from tqdm. It is used to determine
                the progress of the inference.
            client : CommunicationBackend
                Client used to send the message.
            """
            from pipeline_manager_backend_communication.misc_structures import MessageType  # noqa: E501

            progress = int(state["n"] / state["total"] * 100)
            client.send_message(MessageType.PROGRESS,
                                str(progress).encode('UTF-8'))

        logger.set_verbosity(args.verbosity)
        log = logger.get_logger()

        build_status = frontend_builder.build_frontend(
            'server-app',
            workspace_directory=args.workspace_dir,
        )
        if build_status != 0:
            raise RuntimeError('Build error')

        client = CommunicationBackend(args.host, args.port)
        start_server_in_parallel(frontend_path=args.workspace_dir / 'frontend/dist')

        try:
            if args.spec_type == "pipeline":
                dataflow_handler = PipelineHandler(layout_algorithm=args.layout) # noqa E501
            elif args.spec_type == "flow":
                dataflow_handler = KenningFlowHandler(layout_algorithm=args.layout) # noqa E501
            else:
                raise RuntimeError(f"Unrecognized f{args.spec_type} spec_type")

            client.initialize_client()

            callback_percent = Callback('runtime', send_progress, 1.0, client)
            TqdmCallback.register_callback(callback_percent)

            while client.client_socket:
                status, message = client.wait_for_message()
                if status == Status.DATA_READY:
                    message_type, data = message

                    try:
                        return_status, return_message = parse_message(
                            dataflow_handler, message_type,
                            data, args.file_path
                        )
                    except ModuleNotFoundError as e:
                        extras = find_missing_optional_dependency(e.name)
                        error_message = MissingKenningDependencies(
                            name=e.name, path=e.path,
                            optional_dependencies=extras)
                        client.send_message(
                            MessageType.ERROR,
                            bytes(str(error_message), 'utf-8'))
                        continue

                    client.send_message(return_status, return_message)

            TqdmCallback.unregister_callback(callback_percent)
        except ValidationError as ex:
            log.error(f'Failed to load JSON file:\n{ex}')
            client.send_message(
                MessageType.ERROR,
                bytes(f'Failed to load JSON file:\n{ex}', 'utf-8'))
            return 1
        except RuntimeError as ex:
            log.error(f'Server runtime error:\n{ex}')
            client.send_message(
                MessageType.ERROR,
                bytes(f'Server runtime error:\n{ex}', 'utf-8'))
            return 1
        except ConnectionRefusedError as ex:
            log.error(
                f'Could not connect to the Pipeline Manager server: {ex}')
            client.send_message(
                MessageType.ERROR,
                bytes(f'Could not connect to the Pipeline Manager server: {ex}', 'utf-8')) # noqa E501
            return ex.errno
        except Exception as ex:
            log.error(f'Unexpected exception:\n{ex}')
            client.send_message(
                MessageType.ERROR,
                bytes(f'Unexpected exception:\n{ex}', 'utf-8'))
            raise
        finally:
            stop_parallel_server()
        return 0


if __name__ == '__main__':
    sys.exit(PipelineManagerClient.scenario_run())
