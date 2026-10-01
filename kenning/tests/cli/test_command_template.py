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
        """
        Tests if consecutive calls generate unique command type IDs.
        """
        ids = [generate_command_type() for _ in range(5)]

        assert len(set(ids)) == len(ids)

    def test_ids_are_handed_out_in_order(self) -> None:
        """
        Tests if generated command type IDs are increasing.
        """
        ids = [generate_command_type() for _ in range(5)]

        assert ids == sorted(ids)

    def test_the_template_itself_owns_the_first_id(self) -> None:
        """
        Tests if CommandTemplate ID is lower than any newly generated one.
        """
        assert CommandTemplate.ID < generate_command_type()


class TestScenarioIds:
    @pytest.mark.parametrize(
        "command,scenario", sorted(MAP_COMMAND_TO_SCENARIO.items())
    )
    def test_every_scenario_declares_its_own_id(
        self, command: str, scenario: Type[CommandTemplate]
    ) -> None:
        """
        Tests if each scenario defines its own ID attribute.
        """
        assert "ID" in vars(scenario)

    @pytest.mark.parametrize(
        "command,scenario", sorted(MAP_COMMAND_TO_SCENARIO.items())
    )
    def test_no_scenario_falls_back_to_the_template_id(
        self, command: str, scenario: Type[CommandTemplate]
    ) -> None:
        """
        Tests if no scenario reuses the CommandTemplate ID.
        """
        assert scenario.ID != CommandTemplate.ID


class TestConfigureParser:
    def test_a_parser_is_created_when_none_is_given(self) -> None:
        """
        Tests if a new Parser is created when no parser is provided.
        """
        parser, _ = CommandTemplate.configure_parser(command="kenning test")

        assert isinstance(parser, Parser)
        assert parser.prog == "kenning test"

    def test_the_given_parser_is_extended_instead_of_replaced(self) -> None:
        """
        Tests if the provided parser is returned instead of a new one.
        """
        given = argparse.ArgumentParser(add_help=False)

        parser, _ = CommandTemplate.configure_parser(parser=given)

        assert parser is given

    def test_help_and_verbosity_land_in_the_default_group(self) -> None:
        """
        Tests if help and verbosity flags are added to the default group.
        """
        _, groups = CommandTemplate.configure_parser()

        assert set(groups) == {DEFAULT_GROUP}
        assert set(HELP["flags"]) <= set(flags_of(groups[DEFAULT_GROUP]))
        assert "--verbosity" in flags_of(groups[DEFAULT_GROUP])

    def test_the_groups_registry_is_reused_by_the_next_scenario(self) -> None:
        """
        Tests if passed groups are reused without duplicating them.
        """
        parser, groups = CommandTemplate.configure_parser(
            resolve_conflict=True
        )

        _, next_groups = CommandTemplate.configure_parser(
            parser=parser, groups=groups
        )

        assert next_groups is groups
        assert next_groups[DEFAULT_GROUP] is groups[DEFAULT_GROUP]
        assert group_titles(parser).count(DEFAULT_GROUP) == 1


class TestAddGroups:
    def test_a_list_creates_a_group_per_title(self) -> None:
        """
        Tests if a group is created for each title from the list.
        """
        parser = argparse.ArgumentParser(add_help=False)

        groups = CommandTemplate.add_groups(parser, None, ["first", "second"])

        assert set(groups) == {"first", "second"}
        assert "first" in group_titles(parser)
        assert "second" in group_titles(parser)

    def test_a_dict_also_sets_the_descriptions(self) -> None:
        """
        Tests if groups passed as dict get their descriptions set.
        """
        parser = argparse.ArgumentParser(add_help=False)

        groups = CommandTemplate.add_groups(
            parser, None, {"first": "some description"}
        )

        assert groups["first"].description == "some description"

    def test_a_missing_registry_starts_an_empty_one(self) -> None:
        """
        Tests if an empty dict is returned when no groups are given.
        """
        parser = argparse.ArgumentParser(add_help=False)

        groups = CommandTemplate.add_groups(parser, None, [])

        assert groups == {}

    def test_new_titles_extend_an_existing_registry(self) -> None:
        """
        Tests if new groups are added to the existing groups dict.
        """
        parser = argparse.ArgumentParser(add_help=False)
        groups = CommandTemplate.add_groups(parser, None, ["first"])

        groups = CommandTemplate.add_groups(parser, groups, ["second"])

        assert set(groups) == {"first", "second"}

    def test_a_known_title_is_not_created_twice(self) -> None:
        """
        Tests if an already existing group is not created again.
        """
        parser = argparse.ArgumentParser(add_help=False)
        groups = CommandTemplate.add_groups(parser, None, ["first"])
        created = groups["first"]

        groups = CommandTemplate.add_groups(parser, groups, ["first"])

        assert groups["first"] is created
        assert group_titles(parser).count("first") == 1

    def test_arguments_stay_in_the_group_they_were_added_to(self) -> None:
        """
        Tests if arguments are assigned only to their own group.
        """
        parser = argparse.ArgumentParser(add_help=False)
        groups = CommandTemplate.add_groups(parser, None, ["first", "second"])

        groups["first"].add_argument("--in-first")
        groups["second"].add_argument("--in-second")

        assert flags_of(groups["first"]) == ["--in-first"]
        assert flags_of(groups["second"]) == ["--in-second"]
