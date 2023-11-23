#!/usr/bin/env python

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
A script for connecting Kenning with Pipeline Manager server.
"""

import argparse
import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from jsonschema.exceptions import ValidationError

from kenning.cli.command_template import (
    GROUP_SCHEMA,
    VISUAL_EDITOR,
    ArgumentsGroups,
    CommandTemplate,
)
from kenning.core.measurements import MeasurementsCollector
from kenning.utils.logger import Callback, KLogger, TqdmCallback


class PipelineManagerShutdownException(Exception):
    """
    Raised when Pipeline Manager ends its work.
    """

    pass


class PipelineManagerClient(CommandTemplate):
    """
    Command template for Visual Editor.
    """

    parse_all = True
    description = __doc__.split("\n\n")[0]
    specification = None

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
            help="Path where results of model benchmarking will be stored (pipeline mode only)",  # noqa: E501
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

    @staticmethod
    def run(args: argparse.Namespace, **kwargs):
        from pipeline_manager import frontend_builder
        from pipeline_manager.backend.run_in_parallel import (
            start_server_in_parallel,
            stop_parallel_server,
        )
        from pipeline_manager_backend_communication.communication_backend import (  # noqa: E501
            CommunicationBackend,
        )
        from pipeline_manager_backend_communication.misc_structures import (
            MessageType,
        )

        from kenning.pipeline_manager.core import BaseDataflowHandler
        from kenning.pipeline_manager.flow_handler import KenningFlowHandler
        from kenning.pipeline_manager.pipeline_handler import PipelineHandler

        class PipelineManagerRPC:
            def __init__(
                self,
                dataflow_handler: BaseDataflowHandler,
                output_file_path: Path,
            ):
                self.dataflow_handler = dataflow_handler
                self.output_file_path = output_file_path

            def dataflow_import(
                self, external_application_dataflow: Dict
            ) -> Dict:
                """
                Imports Kenning scenario to Pipeline Manager.

                Parameters
                ----------
                external_application_dataflow: Dict
                    Scenario in Kenning format

                Returns
                -------
                Dict
                    Pipeline Manager graph representing the scenario
                """
                graph_repr = self.dataflow_handler.create_dataflow(
                    external_application_dataflow
                )
                return {"type": MessageType.OK.value, "content": graph_repr}

            def specification_get(self) -> Dict:
                """
                Returns the specification of available Kenning classes.

                Returns
                -------
                Dict
                    Method's response
                """
                if not PipelineManagerClient.specification:
                    KLogger.info(
                        "SpecificationBuilder: Generating specification..."
                    )
                    PipelineManagerClient.specification = (
                        self.dataflow_handler.get_specification(
                            workspace_dir=args.workspace_dir,
                            spec_save_path=args.save_specification_path,
                        )
                    )
                return {
                    "type": MessageType.OK.value,
                    "content": PipelineManagerClient.specification,
                }

            async def dataflow_run(self, dataflow: Dict) -> Dict:
                """
                Runs the given Kenning scenario.

                Parameters
                ----------
                dataflow : Dict
                    Content of the request.

                Returns
                -------
                Dict
                    Method's response
                """
                status, msg = self.dataflow_handler.parse_dataflow(dataflow)
                if not status:
                    return {"type": MessageType.ERROR.value, "content": msg}
                try:
                    runner = self.dataflow_handler.parse_json(msg)
                    MeasurementsCollector.clear()

                    def dataflow_runner(runner):
                        self.dataflow_handler.run_dataflow(
                            runner, self.output_file_path
                        )

                    task = asyncio.to_thread(dataflow_runner, runner)
                    await asyncio.gather(task)
                except Exception as ex:
                    KLogger.error(ex, stack_info=True)
                    return {
                        "type": MessageType.ERROR.value,
                        "content": str(ex),
                    }
                return {
                    "type": MessageType.OK.value,
                    "content": f"Successfully finished processing. Measurements are saved in {self.output_file_path}",  # noqa: E501
                }

            def dataflow_validate(self, dataflow: Dict) -> Dict:
                """
                Validates the graph in terms of compatibility of classes.

                Parameters
                ----------
                dataflow : Dict
                    Current graph representing the scenario.

                Returns
                -------
                Dict
                    Method's response
                """
                status, msg = self.dataflow_handler.parse_dataflow(dataflow)
                if not status:
                    return {"type": MessageType.ERROR.value, "content": msg}
                try:
                    runner = self.dataflow_handler.parse_json(msg)
                    self.dataflow_handler.destroy_dataflow(runner)
                except Exception as ex:
                    KLogger.error(ex, stack_info=True)
                    return {
                        "type": MessageType.ERROR.value,
                        "content": str(ex),
                    }
                return {
                    "type": MessageType.OK.value,
                    "content": "The graph is valid.",
                }

            def dataflow_export(self, dataflow: Dict) -> Dict:
                """
                Export the graph to Kenning's scenario format.

                Parameters
                ----------
                dataflow : Dict
                    Current graph representing the scenario.

                Returns
                -------
                Dict
                    Method's response
                """
                status, msg = self.dataflow_handler.parse_dataflow(dataflow)
                if not status:
                    return {"type": MessageType.ERROR.value, "content": msg}
                try:
                    runner = self.dataflow_handler.parse_json(msg)
                    with open(self.output_file_path, "w") as f:
                        json.dump(msg, f, indent=4)
                    self.dataflow_handler.destroy_dataflow(runner)
                except Exception as ex:
                    KLogger.error(ex, stack_info=True)
                    return {
                        "type": MessageType.ERROR.value,
                        "content": str(ex),
                    }
                return {
                    "type": MessageType.OK.value,
                    "content": f"The graph is saved to {self.output_file_path}.",  # noqa: E501
                }

        default_sigint_handler = signal.getsignal(signal.SIGINT)

        def exit_handler(*args) -> None:
            KLogger.info("Closing connection")
            signal.signal(signal.SIGINT, default_sigint_handler)
            raise PipelineManagerShutdownException()

        signal.signal(signal.SIGINT, exit_handler)

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
            RuntimeError:
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
            )
            if build_status != 0:
                raise RuntimeError("Build error")

        KLogger.set_verbosity(args.verbosity)

        frontend_files_path = args.workspace_dir / "frontend/dist"

        build_frontend(frontend_path=frontend_files_path)

        def send_progress(
            state: Dict,
            client: CommunicationBackend,
            loop: asyncio.base_events.BaseEventLoop,
        ):
            """
            Sends progress message to Pipeline Manager.

            Parameters
            ----------
            state : Dict
                The `format_dict` that comes from tqdm. It is used to determine
                the progress of the inference.
            client : CommunicationBackend
                Client used to send the message.
            loop : asyncio.base_events.BaseEventLoop
                main asyncio loop
            """
            progress = int(state["n"] / state["total"] * 100)

            async def progress_wrapper(client):
                await client.notify("progress_change", {"progress": progress})

            asyncio.run_coroutine_threadsafe(progress_wrapper(client), loop)

        async def run_client():
            server_id = await start_server_in_parallel(
                frontend_path=frontend_files_path
            )
            await asyncio.sleep(1)

            client = CommunicationBackend(host=args.host, port=args.port)

            callback_percent = Callback(
                "runtime", send_progress, 0.5, client, asyncio.get_event_loop()
            )

            try:
                if args.spec_type == "pipeline":
                    dataflow_handler = PipelineHandler(
                        layout_algorithm=args.layout
                    )
                elif args.spec_type == "flow":
                    dataflow_handler = KenningFlowHandler(
                        layout_algorithm=args.layout
                    )
                else:
                    raise RuntimeError(
                        f"Unrecognized f{args.spec_type} spec_type"
                    )

                rpchandler = PipelineManagerRPC(
                    dataflow_handler, args.file_path
                )
                TqdmCallback.register_callback(callback_percent)
                await client.initialize_client(rpchandler)
                await client.start_json_rpc_client()
            except ValidationError as ex:
                KLogger.error(f"Failed to load JSON file:\n{ex}")
                return 1
            except RuntimeError as ex:
                KLogger.error(f"Server runtime error:\n{ex}")
                return 1
            except ConnectionRefusedError as ex:
                KLogger.error(
                    f"Could not connect to the Pipeline Manager server: {ex}"
                )
                return ex.errno
            except PipelineManagerShutdownException:
                KLogger.info("Closing the Visual Editor...")
            except Exception as ex:
                KLogger.error(f"Unexpected exception:\n{ex}")
                raise
            finally:
                await client.disconnect()
                stop_parallel_server(server_id)
                TqdmCallback.unregister_callback(callback_percent)
                signal.signal(signal.SIGINT, default_sigint_handler)
                KLogger.info("Closed the Visual Editor")
            return 0

        return asyncio.run(run_client())


if __name__ == "__main__":
    sys.exit(PipelineManagerClient.scenario_run())
