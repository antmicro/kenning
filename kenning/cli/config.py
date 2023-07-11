# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with Kenning CLI configuration.

It contains specification which command can be used in sequence
and mapping to class extending CommandTemplate.
"""

from typing import Dict, Type

from kenning.cli.command_template import (
    CommandTemplate,
    OPTIMIZE,
    TEST,
    REPORT,
    VISUAL_EDITOR,
    FLOW,
    SERVER,
    FINE_TUNE,
    HELP
)
from kenning.scenarios import (
    inference_tester,
    render_report,
    pipeline_manager_client,
    json_flow_runner,
    inference_server,
    optimization_runner,
)


# Subcommands that can be used in sequence
SEQUENCED_COMMANDS = (OPTIMIZE, TEST, REPORT)
# Subcommands that can be used one at the time
BASIC_COMMANDS = (FLOW, SERVER, VISUAL_EDITOR, FINE_TUNE)
# All available subcommands and help flags
AVAILABLE_COMMANDS = (*SEQUENCED_COMMANDS, *BASIC_COMMANDS, *HELP["flags"])
# Connection between subcommand and its logic (extending CommandTemplate)
MAP_COMMAND_TO_SCENARIO: Dict[str, Type[CommandTemplate]] = {
    OPTIMIZE: inference_tester.InferenceTester,
    TEST: inference_tester.InferenceTester,
    REPORT: render_report.RenderReport,
    VISUAL_EDITOR: pipeline_manager_client.PipelineManagerClient,
    FLOW: json_flow_runner.FlowRunner,
    SERVER: inference_server.InferenceServer,
    FINE_TUNE: optimization_runner.OptimizationRunner,
}
# Name of the subcommand group -- displayed in help message
SUBCOMMANDS = "Subcommands"
