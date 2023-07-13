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
    FINE_TUNE,
    FLOW,
    HELP,
    INFO,
    LIST,
    OPTIMIZE,
    REPORT,
    SERVER,
    TEST,
    VISUAL_EDITOR,
)
from kenning.scenarios import (
    class_info,
    inference_server,
    inference_tester,
    json_flow_runner,
    list_classes,
    optimization_runner,
    pipeline_manager_client,
    render_report,
)


# Subcommands that can be used in sequence
SEQUENCED_COMMANDS = (OPTIMIZE, TEST, REPORT)
# Subcommands that can be used one at the time
BASIC_COMMANDS = (FLOW, SERVER, VISUAL_EDITOR, FINE_TUNE, LIST, INFO)
# All available subcommands and help flags
AVAILABLE_COMMANDS = (*SEQUENCED_COMMANDS, *BASIC_COMMANDS, *HELP["flags"])
# Connection between subcommand and its logic (extending CommandTemplate)
MAP_COMMAND_TO_SCENARIO: Dict[str, Type[CommandTemplate]] = {
    FINE_TUNE: optimization_runner.OptimizationRunner,
    FLOW: json_flow_runner.FlowRunner,
    INFO: class_info.ClassInfoRunner,
    LIST: list_classes.ListClassesRunner,
    OPTIMIZE: inference_tester.InferenceTester,
    REPORT: render_report.RenderReport,
    SERVER: inference_server.InferenceServer,
    TEST: inference_tester.InferenceTester,
    VISUAL_EDITOR: pipeline_manager_client.PipelineManagerClient,
}
# Name of the subcommand group -- displayed in help message
SUBCOMMANDS = "Subcommands"
