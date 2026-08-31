# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from typing import Dict, List, Tuple

import pytest

from kenning.cli.command_template import (
    AUTOML,
    FLOW,
    OPTIMIZE,
    REPORT,
    ROS,
    TEST,
    TRAIN,
)
from kenning.cli.config import (
    BASIC_COMMANDS,
    MAP_COMMAND_TO_SCENARIO,
    SEQUENCED_COMMANDS,
    SUB_DEST_FORM,
    _either,
    _optional,
    _sequence,
    create_subcommands,
    get_all_sequences,
    setup_base_parser,
)
from kenning.core.exceptions import ConfigurationError


class TestGetAllSequences:
    @pytest.mark.parametrize(
        "grammar,expected",
        [
            (None, [()]),
            ([], [()]),
            (TEST, [(TEST,)]),
        ],
        ids=["none", "empty_list", "single_command"],
    )
    def test_terminals(self, grammar, expected: List[Tuple[str, ...]]) -> None:
        assert list(get_all_sequences(grammar)) == expected

    @pytest.mark.parametrize(
        "grammar,expected",
        [
            ([AUTOML, TRAIN], [(AUTOML, TRAIN)]),
            (
                [OPTIMIZE, (TEST, None)],
                [(OPTIMIZE, TEST), (OPTIMIZE,)],
            ),
            (
                [(AUTOML, TRAIN, None), OPTIMIZE],
                [(AUTOML, OPTIMIZE), (TRAIN, OPTIMIZE), (OPTIMIZE,)],
            ),
        ],
    )
    def test_docstring_examples(
        self, grammar, expected: List[Tuple[str, ...]]
    ) -> None:
        assert list(get_all_sequences(grammar)) == expected

    def test_list_is_a_cartesian_product(self) -> None:
        grammar = _sequence(_either(TRAIN, AUTOML), _either(TEST, REPORT))

        assert set(get_all_sequences(grammar)) == {
            (TRAIN, TEST),
            (TRAIN, REPORT),
            (AUTOML, TEST),
            (AUTOML, REPORT),
        }

    def test_tuple_is_a_union(self) -> None:
        grammar = _either(_sequence(TEST, REPORT), OPTIMIZE)

        assert set(get_all_sequences(grammar)) == {
            (TEST, REPORT),
            (OPTIMIZE,),
        }

    def test_optional_command_is_skipped(self) -> None:
        grammar = _sequence(_optional(ROS), OPTIMIZE)

        assert set(get_all_sequences(grammar)) == {
            (ROS, OPTIMIZE),
            (OPTIMIZE,),
        }

    @pytest.mark.parametrize(
        "sequence",
        [
            (OPTIMIZE,),
            (TEST,),
            (REPORT,),
            (OPTIMIZE, TEST),
            (TRAIN, TEST, REPORT),
            (AUTOML, OPTIMIZE, TEST, REPORT),
            (ROS, FLOW),
        ],
    )
    def test_supported_chains_are_expanded(
        self, sequence: Tuple[str, ...]
    ) -> None:
        assert sequence in set(get_all_sequences(SEQUENCED_COMMANDS))

    def test_every_expanded_command_has_a_scenario(self) -> None:
        # every command reachable through the grammar has to be runnable,
        for sequence in get_all_sequences(SEQUENCED_COMMANDS):
            assert isinstance(sequence, tuple)
            for command in sequence:
                assert command in MAP_COMMAND_TO_SCENARIO


class TestCreateSubcommands:
    @staticmethod
    def empty_registries() -> (
        Tuple[
            argparse.ArgumentParser,
            Dict[Tuple[str, ...], argparse.ArgumentParser],
            Dict[Tuple[str, ...], argparse._SubParsersAction],
        ]
    ):
        root = argparse.ArgumentParser(prog="kenning", add_help=False)
        groups = {(): root.add_subparsers(dest=SUB_DEST_FORM.format(0))}
        return root, {}, groups

    def test_missing_root_group_raises(self) -> None:
        with pytest.raises(ConfigurationError):
            create_subcommands((OPTIMIZE,), {}, {})

    def test_empty_sequence_creates_nothing(self) -> None:
        _, parsers, groups = self.empty_registries()

        assert create_subcommands((), parsers, groups) == {}
        assert parsers == {}
        assert set(groups) == {()}

    def test_only_new_prefixes_are_returned(self) -> None:
        _, parsers, groups = self.empty_registries()

        first = create_subcommands((OPTIMIZE,), parsers, groups)
        parsers.update(first)
        second = create_subcommands((OPTIMIZE, TEST), parsers, groups)
        parsers.update(second)

        assert set(first) == {(OPTIMIZE,)}
        assert set(second) == {(OPTIMIZE, TEST)}

    def test_subparsers_group_is_created_once_per_prefix(self) -> None:
        _, parsers, groups = self.empty_registries()

        parsers.update(create_subcommands((OPTIMIZE, TEST), parsers, groups))
        created = groups[(OPTIMIZE,)]
        parsers.update(create_subcommands((OPTIMIZE, REPORT), parsers, groups))

        assert groups[(OPTIMIZE,)] is created


class TestSetupBaseParser:
    def test_every_sequence_has_a_parser(self) -> None:
        _, parsers = setup_base_parser()

        sequences = set(get_all_sequences(SEQUENCED_COMMANDS)) - {()}
        basic_commands = {(command,) for command in BASIC_COMMANDS}

        assert set(parsers) == sequences | basic_commands

    def test_basic_commands_are_added_next_to_the_grammar(self) -> None:
        _, parsers = setup_base_parser()

        for command in BASIC_COMMANDS:
            assert (command,) in parsers

    def test_unknown_arguments_are_left_over(self) -> None:
        parser, _ = setup_base_parser()

        args, rest = parser.parse_known_args([OPTIMIZE, "--not-a-flag"])

        assert getattr(args, SUB_DEST_FORM.format(0)) == OPTIMIZE
        assert "--not-a-flag" in rest

    def test_arguments_are_configured_on_demand(self) -> None:
        _, bare = setup_base_parser()
        _, configured = setup_base_parser(with_arguments=True)

        assert "--json-cfg" not in bare[(OPTIMIZE,)]._option_string_actions
        assert "--json-cfg" in configured[(OPTIMIZE,)]._option_string_actions

    def test_chained_parser_gathers_arguments_of_every_command(self) -> None:
        parser, parsers = setup_base_parser(with_arguments=True)

        assert (
            "--compiler-cls"
            in parsers[(OPTIMIZE, TEST)]._option_string_actions
        )
        assert (
            "--evaluate-unoptimized"
            in parsers[(OPTIMIZE, TEST)]._option_string_actions
        )

        args, rest = parser.parse_known_args(
            [OPTIMIZE, TEST, "--evaluate-unoptimized", "--not-a-flag"]
        )

        assert getattr(args, SUB_DEST_FORM.format(0)) == OPTIMIZE
        assert getattr(args, SUB_DEST_FORM.format(1)) == TEST
        assert args.evaluate_unoptimized is True
        assert rest == ["--not-a-flag"]
