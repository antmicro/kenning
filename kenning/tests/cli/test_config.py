# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

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
    MAP_COMMAND_TO_SCENARIO,
    SEQUENCED_COMMANDS,
    _either,
    _optional,
    _sequence,
    get_all_sequences,
)


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
