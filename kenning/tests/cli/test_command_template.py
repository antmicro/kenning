# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from typing import List, Type

import pytest

from kenning.cli.command_template import (
    DEFAULT_GROUP,
    HELP,
    CommandTemplate,
    generate_command_type,
)
from kenning.cli.config import MAP_COMMAND_TO_SCENARIO
from kenning.cli.parser import Parser


def group_titles(parser: argparse.ArgumentParser) -> List[str]:
    return [group.title for group in parser._action_groups]


def flags_of(group: argparse._ArgumentGroup) -> List[str]:
    return [
        flag
        for action in group._group_actions
        for flag in action.option_strings
    ]


class TestGenerateCommandType:
    def test_consecutive_calls_return_different_ids(self) -> None:
        ids = [generate_command_type() for _ in range(5)]

        assert len(set(ids)) == len(ids)

    def test_ids_are_handed_out_in_order(self) -> None:
        ids = [generate_command_type() for _ in range(5)]

        assert ids == sorted(ids)

    def test_the_template_itself_owns_the_first_id(self) -> None:
        assert CommandTemplate.ID < generate_command_type()


class TestScenarioIds:
    @pytest.mark.parametrize(
        "command,scenario", sorted(MAP_COMMAND_TO_SCENARIO.items())
    )
    def test_every_scenario_declares_its_own_id(
        self, command: str, scenario: Type[CommandTemplate]
    ) -> None:
        assert "ID" in vars(scenario)

    @pytest.mark.parametrize(
        "command,scenario", sorted(MAP_COMMAND_TO_SCENARIO.items())
    )
    def test_no_scenario_falls_back_to_the_template_id(
        self, command: str, scenario: Type[CommandTemplate]
    ) -> None:
        assert scenario.ID != CommandTemplate.ID


class TestConfigureParser:
    def test_a_parser_is_created_when_none_is_given(self) -> None:
        parser, _ = CommandTemplate.configure_parser(command="kenning test")

        assert isinstance(parser, Parser)
        assert parser.prog == "kenning test"

    def test_the_given_parser_is_extended_instead_of_replaced(self) -> None:
        given = argparse.ArgumentParser(add_help=False)

        parser, _ = CommandTemplate.configure_parser(parser=given)

        assert parser is given

    def test_help_and_verbosity_land_in_the_default_group(self) -> None:
        _, groups = CommandTemplate.configure_parser()

        assert set(groups) == {DEFAULT_GROUP}
        assert set(HELP["flags"]) <= set(flags_of(groups[DEFAULT_GROUP]))
        assert "--verbosity" in flags_of(groups[DEFAULT_GROUP])

    def test_the_groups_registry_is_reused_by_the_next_scenario(self) -> None:
        parser, groups = CommandTemplate.configure_parser(
            resolve_conflict=True
        )

        _, next_groups = CommandTemplate.configure_parser(
            parser=parser, groups=groups
        )

        assert next_groups is groups
        assert next_groups[DEFAULT_GROUP] is groups[DEFAULT_GROUP]
        assert group_titles(parser).count(DEFAULT_GROUP) == 1
