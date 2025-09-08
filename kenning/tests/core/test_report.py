# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace
from pathlib import Path

from pytest_mock import MockerFixture

from kenning.core.report import ReportTypes
from kenning.utils.class_loader import (
    ConfigKey,
    objs_from_argparse,
    objs_from_json,
)

USED_KENNING_REPORT_CLASS = "kenning.report.stub_report.StubReport"


class TestReport:
    @staticmethod
    def sample_config_file(
        measurements: list[str], report_name: str, report_types: list[str]
    ) -> dict:
        return {
            "report": {
                "type": USED_KENNING_REPORT_CLASS,
                "parameters": {
                    "measurements": measurements,
                    "report_name": report_name,
                    "report_types": report_types,
                },
            }
        }

    def test_report_parsing_from_json_config(self):
        """
        Tests loading report parameters from json config.
        """
        MEASUREMENTS = [Path("data.json")]
        REPORT_NAME = "Report from config"
        REPORT_TYPES = [
            report_types.value for report_types in list(ReportTypes)
        ]

        json_cfg = TestReport.sample_config_file(
            [x.name for x in MEASUREMENTS], REPORT_NAME, REPORT_TYPES
        )

        objs = objs_from_json(json_cfg, set([ConfigKey.report]), override=None)

        report = objs[ConfigKey.report]

        assert report.measurements == MEASUREMENTS
        assert report.report_name == REPORT_NAME
        assert report.report_types == REPORT_TYPES

    def test_report_parsing_from_args(self, mocker: MockerFixture):
        """
        Tests loading report from command line arguments.
        """
        MEASUREMENTS = [Path("data.json")]
        REPORT_NAME = "Report from args"
        REPORT_TYPES = [
            report_types.value for report_types in list(ReportTypes)
        ]

        mock_args = Namespace(help=False, report_cls=USED_KENNING_REPORT_CLASS)

        not_parsed = [
            "--measurements",
            *[x.name for x in MEASUREMENTS],
            "--report-name",
            REPORT_NAME,
            "--report-types",
            *REPORT_TYPES,
        ]

        mocker.patch(
            "sys.argv",
            ["kenning"],
        )

        objs = objs_from_argparse(
            mock_args, not_parsed, set([ConfigKey.report])
        )

        report = objs[ConfigKey.report]

        assert report.measurements == MEASUREMENTS
        assert report.report_name == REPORT_NAME
        assert report.report_types == REPORT_TYPES

    def test_report_override_using_args(self, mocker: MockerFixture):
        """
        Tests loading report from config file and override
        parameters using arguments from command line.
        """
        MEASUREMENTS = [Path("data.json")]
        REPORT_NAME = "Report from config"
        REPORT_TYPES = [
            report_types.value for report_types in list(ReportTypes)
        ]

        json_cfg = TestReport.sample_config_file(
            MEASUREMENTS, REPORT_NAME, REPORT_TYPES
        )

        ARGS_MEASUREMENTS = [Path("data1.json"), Path("data.json")]
        ARGS_REPORT_NAME = "Report from args"
        ARGS_REPORT_TYPES = [
            report_types.value for report_types in list(ReportTypes)
        ][0:2]

        not_parsed = [
            "--measurements",
            *[x.name for x in ARGS_MEASUREMENTS],
            "--report-name",
            ARGS_REPORT_NAME,
            "--report-types",
            *ARGS_REPORT_TYPES,
        ]

        mocker.patch(
            "sys.argv",
            ["kenning"],
        )

        mock_args = Namespace(help=False)

        objs = objs_from_json(
            json_cfg, set([ConfigKey.report]), override=(mock_args, not_parsed)
        )

        report = objs[ConfigKey.report]

        assert report.measurements == ARGS_MEASUREMENTS
        assert report.report_name == ARGS_REPORT_NAME
        assert report.report_types == ARGS_REPORT_TYPES
