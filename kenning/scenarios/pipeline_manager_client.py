#!/usr/bin/env python

# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A script for connecting Kenning with Pipeline Manager server.
"""

import argparse
import asyncio
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from kenning.cli.command_template import (
    GROUP_SCHEMA,
    VISUAL_EDITOR,
    ArgumentsGroups,
    CommandTemplate,
    generate_command_type,
)
from kenning.core.exceptions import VisualEditorError
from kenning.utils.logger import (
    Callback,
    DuplicateStream,
    KLogger,
    TqdmCallback,
)


class PipelineManagerClient(CommandTemplate):
    """
    Command template for Visual Editor.
    """

    parse_all = True
    description = __doc__.split("\n\n")[0]
    specification = None
    ID = generate_command_type()

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Optional[ArgumentsGroups] = None,
    ) -> Tuple[argparse.ArgumentParser, ArgumentsGroups]:
        parser, groups = super(
            PipelineManagerClient, PipelineManagerClient
        ).configure_parser(parser, command, types, groups)

        ve_group = parser.add_argument_group(
            GROUP_SCHEMA.format(VISUAL_EDITOR)
        )

        ve_group.add_argument(
            "--host",
            type=str,
            help="The address of the Pipeline Manager Server",
            default="127.0.0.1",
        )
        ve_group.add_argument(
            "--port",
            type=int,
            help="The port of the Pipeline Manager Server",
            default=9000,
        )
        ve_group.add_argument(
            "--file-path",
            type=Path,
            help="Path to a directory where results of model benchmarking will be stored (pipeline mode only)",  # noqa: E501
            required=True,
        )
        ve_group.add_argument(
            "--spec-type",
            type=str,
            help="Type of graph that should be represented in a Pipeline Manager - can choose between optimization pipeline or Kenningflow",  # noqa: E501
            choices=("pipeline", "flow"),
            default="pipeline",
        )
        ve_group.add_argument(
            "--layout",
            help="Autolayout algorithm to use in Pipeline Manager",
            type=str,
            default="CytoscapeEngine - breadthfirst",
        )
        ve_group.add_argument(
            "--workspace-dir",
            type=Path,
            help="Directory where the frontend sources should be stored",
            required=True,
        )
        ve_group.add_argument(
            "--save-specification-path",
            type=Path,
            help="Path to save the specification JSON after it has been generated",  # noqa: E501
        )

        return parser, groups

    @classmethod
    def run(cls, args: argparse.Namespace, **kwargs):
        from pipeline_manager import frontend_builder
        from pipeline_manager.backend.run_in_parallel import (
            start_server_in_parallel,
            stop_parallel_server,
        )
        from pipeline_manager_backend_communication.communication_backend import (  # noqa: E501
            CommunicationBackend,
        )

        from kenning.pipeline_manager.flow_handler import KenningFlowHandler
        from kenning.pipeline_manager.pipeline_handler import PipelineHandler
        from kenning.pipeline_manager.rpc_handler import (
            FlowHandlerRPC,
            OptimizationHandlerRPC,
            PipelineManagerRPC,
        )

        def build_frontend(frontend_path: Path) -> None:
            """
            Builds the Pipeline Manager frontend in the selected directory.
            If some files already exist there, building will be skipped.

            Parameters
            ----------
            frontend_path : Path
                The path where the built frontend files should be stored.

            Raises
            ------
            VisualEditorError:
                Raised if frontend building failed
            """
            # provided these files exist, do not build the frontend again
            if (
                frontend_path.exists()
                and (frontend_path / "index.html").exists()
                and (frontend_path / "js").exists()
                and (frontend_path / "css").exists()
            ):
                return

            build_status = frontend_builder.build_frontend(
                build_type="server-app",
                workspace_directory=args.workspace_dir,
                editor_title="Kenning Visual Editor",
                assets_directory=Path(__file__).parent.parent
                / "resources/visual_editor_resources/",
            )
            if build_status != 0:
                raise VisualEditorError("Build error")

        frontend_files_path = args.workspace_dir / "frontend/dist"

        cls.workspace_dir = args.workspace_dir
        cls.save_specification_path = args.save_specification_path

        build_frontend(frontend_path=frontend_files_path)

        def send_progress(
            state: Dict,
            handler: PipelineManagerRPC,
            client: CommunicationBackend,
            loop: asyncio.base_events.BaseEventLoop,
        ) -> None:
            """
            Sends progress message to Pipeline Manager.

            Parameters
            ----------
            state : Dict
                The `format_dict` that comes from tqdm. It is used to determine
                the progress of the inference.
            handler : PipelineManagerRPC
                handler of RPC calls
            client : CommunicationBackend
                Client used to send the message.
            loop : asyncio.base_events.BaseEventLoop
                main asyncio loop

            Returns
            -------
            None
                No return value
            """
            if handler.current_method is None:
                return
            progress = -1
            if state["total"] is not None:
                progress = int(state["n"] / state["total"] * 100)
            asyncio.run_coroutine_threadsafe(
                client.notify(
                    "progress_change",
                    {"progress": progress, "method": handler.current_method},
                ),
                loop,
            )

        async def run_client():
            server_id = await start_server_in_parallel(
                frontend_path=frontend_files_path
            )
            await asyncio.sleep(1)

            client = CommunicationBackend(host=args.host, port=args.port)

            loop = asyncio.get_event_loop()
            DuplicateStream.set_client(client, loop)

            async def exit_handler(signal, loop):
                KLogger.info("Closing the Visual Editor...")
                TqdmCallback.unregister_callback(callback_percent)
                if client.client_transport:
                    client.client_transport.abort()
                stop_parallel_server(server_id)
                KLogger.info("Closed the Visual Editor")

            loop.add_signal_handler(
                signal.SIGINT,
                lambda: asyncio.create_task(exit_handler(signal.SIGINT, loop)),
            )

            try:
                if args.spec_type == "pipeline":
                    dataflow_handler = PipelineHandler(
                        layout_algorithm=args.layout,
                        workspace_dir=cls.workspace_dir.resolve(),
                    )
                    rpchandler = OptimizationHandlerRPC(
                        dataflow_handler, args.file_path, cls, client
                    )
                elif args.spec_type == "flow":
                    dataflow_handler = KenningFlowHandler(
                        layout_algorithm=args.layout,
                        workspace_dir=cls.workspace_dir.resolve(),
                    )
                    rpchandler = FlowHandlerRPC(
                        dataflow_handler, args.file_path, cls, client
                    )
                else:
                    raise VisualEditorError(
                        f"Unrecognized f{args.spec_type} spec_type"
                    )

                callback_percent = Callback(
                    "runtime", send_progress, 0.5, rpchandler, client, loop
                )

                TqdmCallback.register_callback(callback_percent)
                await client.initialize_client(rpchandler)
                await client.start_json_rpc_client()
            except ConnectionRefusedError as ex:
                KLogger.error(
                    f"Could not connect to the Pipeline Manager server: {ex}"
                )
                return ex.errno
            except Exception as ex:
                KLogger.error(f"Unexpected exception:\n{ex}")
                return ex.errno
            return 0

        return asyncio.run(run_client())


if __name__ == "__main__":
    sys.exit(PipelineManagerClient.scenario_run())
