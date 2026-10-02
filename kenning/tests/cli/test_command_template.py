# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import pytest

from kenning.cli.command_template import (
    DEFAULT_GROUP,
    HELP,
    CommandTemplate,
    generate_command_type,
)
from kenning.cli.config import MAP_COMMAND_TO_SCENARIO
from kenning.cli.parser import Parser
from kenning.dispatcher.block_config import (
    BLOCK_CONFIG_PARAMETERS_KEY,
    BLOCK_CONFIGURATIONS_KEY,
    UNAFFILIATED_PARAMETERS_KEY,
    ConfigKey,
)
from kenning.utils.class_loader import classes_from_full_dict_config

KEYS = [ConfigKey.dataset, ConfigKey.platform]


def group_titles(parser: argparse.ArgumentParser) -> List[str]:
    return [group.title for group in parser._action_groups]


def make_args(
    json_cfg: Optional[Path] = None,
    dataset_cls: Optional[str] = None,
    platform_cls: Optional[str] = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        json_cfg=json_cfg,
        help=False,
        verbosity="INFO",
        dataset_cls=dataset_cls,
        platform_cls=platform_cls,
    )


def write_config(tmp_path: Path, config: Dict[str, Any]) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def blocks_of(config: Dict, key: ConfigKey) -> Dict:
    return config[BLOCK_CONFIGURATIONS_KEY][key.value]


def selected_class(config: Dict, key: ConfigKey) -> str:
    return classes_from_full_dict_config(config)[key].__name__


class RecordingScenario(CommandTemplate):
    parse_all = True
    did_run = False

    @staticmethod
    def run(
        args: argparse.Namespace, not_parsed: List[str] = [], **kwargs: Any
    ) -> int:
        RecordingScenario.did_run = True
        return 0


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


class TestParseConfiguration:
    FILE_DATASET = {
        "type": "MagicWandDataset",
        "parameters": {"dataset_root": "./from-file", "batch_size": 4},
    }

    def test_flags_alone_build_the_config(self) -> None:
        """
        Tests if blocks and their flags are taken from CLI without a file.
        """
        config = CommandTemplate.parse_configuration(
            make_args(dataset_cls="MagicWandDataset"),
            ["--dataset-root", "./from-cli"],
            KEYS,
        )

        assert selected_class(config, ConfigKey.dataset) == "MagicWandDataset"
        assert config[UNAFFILIATED_PARAMETERS_KEY]["dataset_root"] == Path(
            "./from-cli"
        )

    def test_file_alone_builds_the_config(self, tmp_path: Path) -> None:
        """
        Tests if blocks and their parameters are taken from the config file.
        """
        cfg = write_config(tmp_path, {"dataset": self.FILE_DATASET})

        config = CommandTemplate.parse_configuration(make_args(cfg), [], KEYS)

        assert selected_class(config, ConfigKey.dataset) == "MagicWandDataset"
        assert (
            blocks_of(config, ConfigKey.dataset)["MagicWandDataset"][
                BLOCK_CONFIG_PARAMETERS_KEY
            ]
            == self.FILE_DATASET["parameters"]
        )

    def test_cli_class_overrides_the_file_class(self, tmp_path: Path) -> None:
        """
        Tests if a block class given in CLI replaces the one from the file.
        """
        cfg = write_config(tmp_path, {"dataset": self.FILE_DATASET})

        config = CommandTemplate.parse_configuration(
            make_args(cfg, dataset_cls="AnomalyDetectionDataset"), [], KEYS
        )

        assert "MagicWandDataset" not in blocks_of(config, ConfigKey.dataset)
        assert (
            selected_class(config, ConfigKey.dataset)
            == "AnomalyDetectionDataset"
        )

    def test_flags_of_the_cli_class_are_accepted(self, tmp_path: Path) -> None:
        """
        Tests if flags of the CLI block class are parsed, when the file
        chooses a different class.
        """
        cfg = write_config(tmp_path, {"dataset": self.FILE_DATASET})

        config = CommandTemplate.parse_configuration(
            make_args(cfg, dataset_cls="AnomalyDetectionDataset"),
            ["--label-type", "some-label"],
            KEYS,
        )

        assert (
            config[UNAFFILIATED_PARAMETERS_KEY]["label_type"] == "some-label"
        )

    def test_same_class_in_cli_keeps_file_parameters(
        self, tmp_path: Path
    ) -> None:
        """
        Tests if file parameters are kept when CLI chooses the same class.
        """
        cfg = write_config(tmp_path, {"dataset": self.FILE_DATASET})

        config = CommandTemplate.parse_configuration(
            make_args(cfg, dataset_cls="kenning.datasets.MagicWandDataset"),
            [],
            KEYS,
        )

        assert selected_class(config, ConfigKey.dataset) == "MagicWandDataset"
        assert (
            blocks_of(config, ConfigKey.dataset)["MagicWandDataset"][
                BLOCK_CONFIG_PARAMETERS_KEY
            ]
            == self.FILE_DATASET["parameters"]
        )

    def test_file_block_takes_precedence_over_the_default(
        self, tmp_path: Path
    ) -> None:
        """
        Tests if a block from the file is used instead of the default one.
        """
        cfg = write_config(tmp_path, {"platform": {"type": "CUDAPlatform"}})

        config = CommandTemplate.parse_configuration(make_args(cfg), [], KEYS)

        assert selected_class(config, ConfigKey.platform) == "CUDAPlatform"

    def test_cli_block_takes_precedence_over_the_default(self) -> None:
        """
        Tests if a block from CLI is used instead of the default one.
        """
        config = CommandTemplate.parse_configuration(
            make_args(platform_cls="CUDAPlatform"), [], KEYS
        )

        assert selected_class(config, ConfigKey.platform) == "CUDAPlatform"


class TestScenarioRun:
    @pytest.fixture(autouse=True)
    def reset_did_run(self) -> None:
        RecordingScenario.did_run = False

    def test_the_scenario_runs_without_help(self) -> None:
        """
        Tests if the scenario is run when help is not requested.
        """
        assert RecordingScenario.scenario_run(["kenning"]) == 0
        assert RecordingScenario.did_run

    @pytest.mark.parametrize("flag", HELP["flags"])
    def test_the_scenario_is_not_run_with_help(
        self, flag: str, capsys: pytest.CaptureFixture
    ) -> None:
        """
        Tests if help is printed instead of running the scenario.
        """
        assert RecordingScenario.scenario_run(["kenning", flag]) is None
        assert not RecordingScenario.did_run
        assert "usage" in capsys.readouterr().out.lower()
