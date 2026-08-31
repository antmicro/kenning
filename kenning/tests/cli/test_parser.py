# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from typing import List

import pytest

from kenning.cli.command_template import (
    AUTOML,
    OPTIMIZE,
    REPORT,
    TEST,
)
from kenning.cli.parser import (
    HELP_FLAGS,
    USED_SUBCOMMANDS,
    Parser,
    ParserHelpException,
    get_used_subcommands,
)


def make_parser(prog: str = "kenning test", **kwargs) -> Parser:
    parser = Parser(prog, add_help=False, **kwargs)
    parser.add_argument(*HELP_FLAGS, action="store_true", help="Show help")
    return parser


class TestGetUsedSubcommands:
    @pytest.mark.parametrize(
        "args,expected",
        [
            (argparse.Namespace(), []),
            (argparse.Namespace(json_cfg="config.json"), []),
            (argparse.Namespace(**{USED_SUBCOMMANDS: []}), []),
            (argparse.Namespace(**{USED_SUBCOMMANDS: [TEST]}), [TEST]),
            (
                argparse.Namespace(**{USED_SUBCOMMANDS: [OPTIMIZE, TEST]}),
                [OPTIMIZE, TEST],
            ),
            (
                argparse.Namespace(
                    **{USED_SUBCOMMANDS: [AUTOML, OPTIMIZE, TEST, REPORT]}
                ),
                [AUTOML, OPTIMIZE, TEST, REPORT],
            ),
        ],
        ids=[
            "nothing_stored",
            "only_unrelated_arguments",
            "empty_sequence",
            "single_subcommand",
            "two_subcommands",
            "full_chain",
        ],
    )
    def test_returns_the_stored_subcommands(
        self, args: argparse.Namespace, expected: List[str]
    ) -> None:
        assert get_used_subcommands(args) == expected


class TestParserHelpException:
    def test_print_includes_the_carried_parser(self, capsys) -> None:
        base = make_parser("kenning test")
        carried = argparse.ArgumentParser("kenning runtime", add_help=False)
        carried.add_argument("--carried-flag")

        ParserHelpException(carried).print(base)

        out = capsys.readouterr().out
        assert "--carried-flag" in out
        assert "kenning runtime" in out

    def test_print_reports_the_error_and_exits_with_zero(self, capsys) -> None:
        parser = make_parser("kenning test")

        with pytest.raises(SystemExit) as exit_info:
            ParserHelpException(error="missing --json-cfg").print(parser)

        assert exit_info.value.code == 0
        assert "missing --json-cfg" in capsys.readouterr().err


class TestParserError:
    def test_early_exit_uses_status_zero(self, capsys) -> None:
        parser = make_parser("kenning test")

        with pytest.raises(SystemExit) as exit_info:
            parser.error("done", early_exit=True)

        assert exit_info.value.code == 0
        assert "kenning test: error: done" in capsys.readouterr().err

    def test_help_flag_raises_instead_of_exiting(self) -> None:
        parser = make_parser()
        parser.parse_known_args(["--help"])

        with pytest.raises(ParserHelpException) as exception_info:
            parser.error("boom")

        assert exception_info.value.parser is parser
        assert exception_info.value.error == "boom"

    def test_without_the_help_flag_it_exits_with_two(self, capsys) -> None:
        parser = make_parser("kenning test")
        parser.parse_known_args([])

        with pytest.raises(SystemExit) as exit_info:
            parser.error("boom")

        assert exit_info.value.code == 2
        assert "kenning test: error: boom" in capsys.readouterr().err

    def test_help_flag_is_read_from_the_parsed_arguments(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["kenning", "test"])
        parser = make_parser()
        parser.parse_known_args(["--help"])

        with pytest.raises(ParserHelpException):
            parser.error("boom")


class TestParserParseArgs:
    def test_unrecognized_arguments_raise_when_help_is_requested(self) -> None:
        parser = make_parser()

        with pytest.raises(ParserHelpException) as exception_info:
            parser.parse_args(["--help", "--not-a-flag"])

        assert exception_info.value.error == (
            "unrecognized arguments: --not-a-flag"
        )


class TestHelpWinsOverErrors:
    def test_help_wins_over_missing_required_arguments(self) -> None:
        parser = make_parser()
        parser.add_argument("--json-cfg", required=True)

        with pytest.raises(ParserHelpException) as exception_info:
            parser.parse_args(["--help"])

        assert exception_info.value.error == (
            "the following arguments are required: --json-cfg"
        )

    def test_help_after_a_malformed_flag_still_wins(self) -> None:
        parser = make_parser()
        parser.add_argument("--json-cfg")

        with pytest.raises(ParserHelpException) as exception_info:
            parser.parse_args(["--json-cfg", "--help"])

        assert "expected one argument" in exception_info.value.error
