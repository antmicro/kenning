# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Type

import pytest

from kenning.cli.command_template import (
    CommandTemplate,
    generate_command_type,
)
from kenning.cli.config import MAP_COMMAND_TO_SCENARIO


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
