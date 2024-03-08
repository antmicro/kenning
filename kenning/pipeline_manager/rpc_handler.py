# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Handlers for RPC for Pipeline Manager.
"""

import asyncio
import datetime
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

from jsonrpc.exceptions import JSONRPCDispatchException
from jsonschema.exceptions import ValidationError
from pipeline_manager_backend_communication.misc_structures import (
    MessageType,
)
from pipeline_manager_backend_communication.utils import (
    convert_message_to_string,
)

from kenning.core.drawing import (
    IMMATERIAL_COLORS,
    RED_GREEN_CMAP,
    SERVIS_PLOT_OPTIONS,
    choose_theme,
)
from kenning.core.measurements import MeasurementsCollector
from kenning.pipeline_manager.core import BaseDataflowHandler
from kenning.scenarios.render_report import (
    generate_html_report,
    generate_report,
    load_measurements_for_report,
)
from kenning.utils.class_loader import get_command
from kenning.utils.logger import KLogger


class PipelineManagerRPC(ABC):
    """
    General class containing RPC callbacks for Pipeline Manager.
    """

    def __init__(
        self,
        dataflow_handler: BaseDataflowHandler,
        output_file_path: Path,
        pipeline_manager_client,
        client,
    ):
        self.dataflow_handler = dataflow_handler
        self.output_file_path = output_file_path
        self.current_task = None
        self.current_method = None
        self.current_task_lock = asyncio.Lock()
        self.pipeline_manager_client = pipeline_manager_client
        self.client = client
        self.filename = None

    @abstractmethod
    def app_capabilities_get(self):
        ...

    @abstractmethod
    def get_navbar_actions(self):
        ...

    async def dataflow_import(
        self,
        external_application_dataflow: Dict,
        mime: str,
        base64: bool,
    ) -> Dict:
        """
        Imports Kenning scenario to Pipeline Manager.

        Parameters
        ----------
        external_application_dataflow: Dict
            Scenario in Kenning format
        mime: str
            MIME type of the received file
        base64: bool
            Tells whether file is in byte64 format

        Returns
        -------
        Dict
            Pipeline Manager graph representing the scenario
        """
        external_application_dataflow = json.loads(
            convert_message_to_string(
                message=external_application_dataflow,
                mime=mime,
                base64=base64,
            )
        )
        async with self.current_task_lock:
            if self.current_task is not None:
                return {
                    "type": MessageType.ERROR.value,
                    "content": f"Can't import pipeline - task {self.current_task} is running",  # noqa: E501
                }
            self.current_task = "importing pipeline"
        try:
            KLogger.info("Importing scenario...")
            graph_repr = self.dataflow_handler.create_dataflow(
                external_application_dataflow
            )
            KLogger.info("Imported scenario.")
            return {
                "type": MessageType.OK.value,
                "content": graph_repr,
            }
        except (
            asyncio.exceptions.CancelledError,
            JSONRPCDispatchException,
        ):
            KLogger.warning("Cancelling import job")
        except (ValidationError, RuntimeError) as ex:
            KLogger.error(f"Failed to load JSON file:\n{ex}")
            return {
                "type": MessageType.ERROR.value,
                "content": f"Failed to load scenario:\n{ex}",  # noqa: E501
            }
        finally:
            async with self.current_task_lock:
                self.current_task = None
                self.current_method = None

    async def dataflow_export(self, dataflow: Dict) -> Dict:
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
        async with self.current_task_lock:
            if self.current_task is not None:
                return {
                    "type": MessageType.ERROR.value,
                    "content": f"Can't export pipeline - task {self.current_task} is running",  # noqa: E501
                }
            self.current_task = "exporting pipeline"
        try:
            KLogger.info("Exporting scenario...")
            status, msg = self.dataflow_handler.parse_dataflow(dataflow)
            if not status:
                return {
                    "type": MessageType.ERROR.value,
                    "content": msg,
                }
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
            KLogger.info(f"Saved scenario in {self.output_file_path}.")
            return {
                "type": MessageType.OK.value,
                "content": f"The graph is saved to {self.output_file_path}.",  # noqa: E501
            }
        except (
            asyncio.exceptions.CancelledError,
            JSONRPCDispatchException,
        ):
            KLogger.warning("Cancelling export job")
        except (ValidationError, RuntimeError) as ex:
            KLogger.error(f"Failed to save the scenario:\n{ex}")
            return {
                "type": MessageType.ERROR.value,
                "content": f"Failed to save the scenario:\n{ex}",  # noqa: E501
            }
        finally:
            async with self.current_task_lock:
                self.current_task = None
                self.current_method = None

    async def specification_get(self) -> Dict:
        """
        Returns the specification of available Kenning classes.

        Returns
        -------
        Dict
            Method's response
        """
        try:
            if not self.pipeline_manager_client.specification:
                KLogger.info(
                    "SpecificationBuilder: Generating specification..."
                )
                self.pipeline_manager_client.specification = self.dataflow_handler.get_specification(  # noqa: E501
                    workspace_dir=self.pipeline_manager_client.workspace_dir,
                    actions=self.get_navbar_actions(),
                    spec_save_path=self.pipeline_manager_client.save_specification_path,
                )
                KLogger.info("SpecificationBuilder: Generated specification.")
            return {
                "type": MessageType.OK.value,
                "content": self.pipeline_manager_client.specification,
            }
        except (
            asyncio.exceptions.CancelledError,
            JSONRPCDispatchException,
        ):
            KLogger.warning("Cancelling specification building")
        except (ValidationError, RuntimeError) as ex:
            KLogger.error(f"Failed to generate specification:\n{ex}")
            return {
                "type": MessageType.ERROR.value,
                "content": f"Failed to generate specification:\n{ex}",  # noqa: E501
            }

    async def dataflow_validate(self, dataflow: Dict) -> Dict:
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
        async with self.current_task_lock:
            if self.current_task is not None:
                return {
                    "type": MessageType.ERROR.value,
                    "content": f"Can't validate pipeline - task {self.current_task} is running",  # noqa: E501
                }
            self.current_task = "validating pipeline"
            self.current_method = "dataflow_validate"
            await self.client.notify(
                "progress_change",
                {
                    "progress": -1,
                    "method": "dataflow_validate",
                },
            )
        try:
            KLogger.info("SpecificationBuilder: Validating specification...")
            status, msg = self.dataflow_handler.parse_dataflow(dataflow)
            if not status:
                KLogger.info("SpecificationBuilder: Specification is invalid.")
                return {
                    "type": MessageType.ERROR.value,
                    "content": msg,
                }
            try:
                runner = self.dataflow_handler.parse_json(msg)
                self.dataflow_handler.destroy_dataflow(runner)
            except Exception as ex:
                KLogger.error(ex, stack_info=True)
                return {
                    "type": MessageType.ERROR.value,
                    "content": str(ex),
                }
            async with self.current_task_lock:
                self.current_task = None
            KLogger.info("SpecificationBuilder: Specification is valid.")
            return {
                "type": MessageType.OK.value,
                "content": "The graph is valid.",
            }
        except (
            asyncio.exceptions.CancelledError,
            JSONRPCDispatchException,
        ):
            KLogger.warning("Cancelling validate job")
        except (ValidationError, RuntimeError) as ex:
            KLogger.error(f"Validation error:\n{ex}")
            return {
                "type": MessageType.ERROR.value,
                "content": f"Validation error:\n{ex}",  # noqa: E501
            }
        finally:
            async with self.current_task_lock:
                self.current_task = None
                self.current_method = None

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
        async with self.current_task_lock:
            if self.current_task is not None:
                return {
                    "type": MessageType.ERROR.value,
                    "content": f"Can't run pipeline - task {self.current_task} is running",  # noqa: E501
                }
            self.current_task = "running pipeline"
            self.current_method = "dataflow_run"
            await self.client.notify(
                "progress_change",
                {
                    "progress": -1,
                    "method": "dataflow_run",
                },
            )
        try:
            KLogger.info("Running scenario...")
            status, msg = self.dataflow_handler.parse_dataflow(dataflow)
            if not status:
                return {
                    "type": MessageType.ERROR.value,
                    "content": msg,
                }
            try:
                runner = self.dataflow_handler.parse_json(msg)
                MeasurementsCollector.clear()

                if not self.output_file_path.exists():
                    self.output_file_path.mkdir(parents=True, exist_ok=True)
                current_time = datetime.datetime.now()
                timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                self.filename = f"run_{timestamp}.json"

                def dataflow_runner(runner):
                    self.dataflow_handler.run_dataflow(
                        runner, self.output_file_path / self.filename
                    )
                    KLogger.info("Finished run")

                runner_coro = asyncio.to_thread(dataflow_runner, runner)
                runner_task = asyncio.create_task(runner_coro)
                await runner_task
            except Exception as ex:
                KLogger.error(ex, stack_info=True)
                return {
                    "type": MessageType.ERROR.value,
                    "content": str(ex),
                }
            except asyncio.exceptions.CancelledError:
                if hasattr(runner, "should_cancel"):
                    runner.should_cancel = True
                if hasattr(runner, "model_wrapper"):
                    runner.model_wrapper.should_cancel = True
            KLogger.info("Finished running scenario.")
            return {
                "type": MessageType.OK.value,
                "content": f"Successfully finished processing. Measurements are saved in {self.output_file_path}",  # noqa: E501
            }
        except (
            asyncio.exceptions.CancelledError,
            JSONRPCDispatchException,
        ):
            KLogger.warning("Cancelling run job")
        except (ValidationError, RuntimeError) as ex:
            KLogger.error(f"Failed to run the pipeline:\n{ex}")
            return {
                "type": MessageType.ERROR.value,
                "content": f"Failed to run the pipeline:\n{ex}",  # noqa: E501
            }
        finally:
            async with self.current_task_lock:
                self.current_task = None
                self.current_method = None


class OptimizationHandlerRPC(PipelineManagerRPC):
    """
    Pipeline Manager RPC handlers for compilation and evaluation flow.
    """

    def app_capabilities_get(self):
        return {"stoppable_methods": []}

    def get_navbar_actions(self):
        return [
            {
                "name": "Evaluate",
                "iconName": "Run",
                "procedureName": "dataflow_run",
            },
            {
                "name": "Validate",
                "iconName": "Validate",
                "procedureName": "dataflow_validate",
            },
            {
                "name": "Optimize",
                "iconName": "optimize.svg",
                "procedureName": "custom_dataflow_optimize",
            },
            {
                "name": "Report",
                "iconName": "report.svg",
                "procedureName": "custom_dataflow_report",
            },
        ]

    async def custom_dataflow_optimize(self, dataflow: Dict) -> Dict:
        async with self.current_task_lock:
            if self.current_task is not None:
                return {
                    "type": MessageType.ERROR.value,
                    "content": f"Can't create report - task {self.current_task} is running",  # noqa: E501
                }
            self.current_task = "optimizing"
            status, msg = self.dataflow_handler.parse_dataflow(dataflow)
            if not status:
                return {
                    "type": MessageType.ERROR.value,
                    "content": msg,
                }
            self.current_method = "custom_dataflow_optimize"
            await self.client.notify(
                "progress_change",
                {
                    "progress": -1,
                    "method": "custom_dataflow_optimize",
                },
            )
        try:
            KLogger.info("Optimizing model...")
            runner = self.dataflow_handler.parse_json(msg)

            def dataflow_optimizer(runner):
                self.dataflow_handler.optimize_dataflow(runner)
                KLogger.info("Optimized model.")

            runner_coro = asyncio.to_thread(dataflow_optimizer, runner)
            runner_task = asyncio.create_task(runner_coro)
            await runner_task
            return {
                "type": MessageType.OK.value,
                "content": "Model compiled successfully.",
            }
        except Exception as ex:
            KLogger.error(ex, stack_info=True)
            return {
                "type": MessageType.ERROR.value,
                "content": str(ex),
            }
        except (
            asyncio.exceptions.CancelledError,
            JSONRPCDispatchException,
        ):
            KLogger.warning("Cancelling optimizing job")
        except Exception as ex:
            KLogger.error(f"Optimizing error:\n{ex}")
            return {
                "type": MessageType.ERROR.value,
                "content": f"Optimizing error:\n{ex}",
            }
        finally:
            async with self.current_task_lock:
                self.current_task = None
                self.current_method = None

    async def custom_dataflow_report(self, dataflow: Dict) -> Dict:
        async with self.current_task_lock:
            if self.current_task is not None:
                return {
                    "type": MessageType.ERROR.value,
                    "content": f"Can't create report - task {self.current_task} is running",  # noqa: E501
                }
            self.current_task = "reporting"
            self.current_method = "custom_dataflow_report"
            await self.client.notify(
                "progress_change",
                {
                    "progress": -1,
                    "method": "custom_dataflow_report",
                },
            )
        try:
            if self.filename is None:
                return {
                    "type": MessageType.ERROR.value,
                    "content": "Run evaluation before generating a report",
                }
            KLogger.info("Generating report...")
            command = get_command()
            measurementsdata, report_types = load_measurements_for_report(
                measurements_files=[self.output_file_path / self.filename],
                model_names=None,
                skip_unoptimized_model=True,
                report_types=None,
            )
            SERVIS_PLOT_OPTIONS["colormap"] = IMMATERIAL_COLORS
            cmap = RED_GREEN_CMAP
            colors = IMMATERIAL_COLORS
            output_path = self.output_file_path.parent / "report"
            output_path_html = self.output_file_path.parent / "report_html"
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
            if not output_path_html.exists():
                output_path_html.mkdir(parents=True, exist_ok=True)
            if not (output_path / "imgs").exists():
                (output_path / "imgs").mkdir(parents=True, exist_ok=True)
            with choose_theme(
                custom_bokeh_theme=True, custom_matplotlib_theme=True
            ):
                generate_report(
                    report_name="Pipeline Manager Run Report",
                    data=measurementsdata,
                    outputpath=output_path / "report.md",
                    imgdir=output_path / "imgs",
                    report_types=report_types,
                    root_dir=output_path,
                    image_formats={"png", "html"},
                    command=command,
                    cmap=cmap,
                    colors=colors,
                )
                generate_html_report(
                    output_path / "report.md", output_path_html, False
                )
            import webbrowser

            KLogger.info("Generated report.")
            webbrowser.open(
                f"file://{(output_path_html / 'report.html').resolve()}", new=2
            )
            return {
                "type": MessageType.OK.value,
                "content": "The report is generated.",
            }

        except (
            asyncio.exceptions.CancelledError,
            JSONRPCDispatchException,
        ):
            KLogger.warning("Cancelling reporting job")
        except Exception as ex:
            KLogger.error(f"Reporting error:\n{ex}")
            return {
                "type": MessageType.ERROR.value,
                "content": f"Reporting error:\n{ex}",
            }
        finally:
            async with self.current_task_lock:
                self.current_task = None
                self.current_method = None


class FlowHandlerRPC(PipelineManagerRPC):
    """
    Pipeline Manager RPC callbacks for Kenning Flow applications.
    """

    def app_capabilities_get(self):
        return {"stoppable_methods": []}

    def get_navbar_actions(self):
        return [
            {
                "name": "Run",
                "iconName": "Run",
                "procedureName": "dataflow_run",
            },
            {
                "name": "Validate",
                "iconName": "Validate",
                "procedureName": "dataflow_validate",
            },
        ]
