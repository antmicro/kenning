# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import json
from argparse import ArgumentError
from pathlib import Path

import pytest

from kenning.core.report import Report, ReportTypes
from kenning.utils.zpl_suffix import ZplSuffix


class TestDeduceReportTypes:
    @pytest.mark.parametrize(
        "measurements,expected",
        [
            (
                {"eval_confusion_matrix": [[1, 0], [0, 1]]},
                ReportTypes.CLASSIFICATION,
            ),
            ({"metrics_per_threshold": {}}, ReportTypes.ANOMALY),
            ({"eval_gtcount/person": 1}, ReportTypes.DETECTION),
            ({"rouge1": 1.0}, ReportTypes.TEXT_SUMMARIZATION),
            (
                {"regression_true": [1.0], "regression_pred": [1.0]},
                ReportTypes.REGRESSION,
            ),
            ({"depth_true": [1.0]}, ReportTypes.DEPTH_ESTIMATION),
            ({"target_inference_step": [1.0]}, ReportTypes.PERFORMANCE),
            ({"protocol_inference_step": [1.0]}, ReportTypes.PERFORMANCE),
            ({"opcode_counters": {"add": 1}}, ReportTypes.RENODE),
            ({"tokens": [1, 2]}, ReportTypes.LLM_PERFORMANCE),
        ],
    )
    def test_single_report_type_is_deduced(
        self, measurements: dict, expected: ReportTypes
    ):
        """
        Tests that every report type is deduced from the measurement
        entry which is characteristic for it.
        """
        report_types = Report.deduce_report_types([measurements], None)

        assert expected in report_types

    def test_multiple_report_types_are_deduced(self):
        """
        Tests that all matching types are deduced from single measurements.
        """
        report_types = Report.deduce_report_types(
            [
                {
                    "eval_confusion_matrix": [[1, 0], [0, 1]],
                    "target_inference_step": [1.0],
                }
            ],
            None,
        )

        assert ReportTypes.CLASSIFICATION in report_types
        assert ReportTypes.PERFORMANCE in report_types

    def test_no_report_type_suitable_for_all_measurements(self):
        """
        Tests that no report type is deduced when none of them is
        compatible with all measurements.
        """
        report_types = Report.deduce_report_types(
            [
                {"eval_confusion_matrix": [[1, 0], [0, 1]]},
                {"target_inference_step": [1.0]},
            ],
            None,
        )

        assert report_types == []

    def test_zephyr_traces_deduced_from_trace_file(self, tmp_path: Path):
        """
        Tests that the Zephyr traces type is deduced when a trace file
        accompanies the measurements file.
        """
        measurements_path = tmp_path / "measurements.json"
        ZplSuffix.TRACE_JSON._get_path_with_suffix(measurements_path).touch()

        report_types = Report.deduce_report_types(
            [{"target_inference_step": [1.0]}], measurements_path
        )

        assert ReportTypes.ZEPHYR_TRACES in report_types

    def test_zephyr_traces_not_deduced_without_trace_file(
        self, tmp_path: Path
    ):
        """
        Tests that the Zephyr traces type is not deduced when there is no
        trace file next to the measurements file.
        """
        report_types = Report.deduce_report_types(
            [{"target_inference_step": [1.0]}],
            tmp_path / "measurements.json",
        )

        assert ReportTypes.ZEPHYR_TRACES not in report_types


class TestDeduceReportName:
    def test_name_for_single_model_and_type(self):
        """
        Tests the name generated for a single model and a single type.
        """
        name = Report.deduce_report_name(
            [{"model_name": "model"}], ["performance"]
        )

        assert name == "Performance of model"

    def test_name_for_single_model_and_multiple_types(self):
        """
        Tests that all report types are listed in the name.
        """
        name = Report.deduce_report_name(
            [{"model_name": "model"}], ["performance", "classification"]
        )

        assert name == "Performance and classification of model"


class TestLoadMeasurementsForReport:
    @pytest.fixture
    def measurements_path(self, tmp_path: Path) -> Path:
        path = tmp_path / "measurements.json"
        path.write_text(
            json.dumps({"model_name": "model", "target_inference_step": [1.0]})
        )
        return path

    def test_measurements_are_loaded_and_types_deduced(
        self, measurements_path: Path
    ):
        """
        Tests that measurements are read from disk and their report types
        are deduced when none are given.
        """
        data, report_types, automl_stats = Report.load_measurements_for_report(
            [str(measurements_path)], skip_unoptimized_model=True
        )

        assert len(data) == 1
        assert data[0]["model_name"] == "model"
        assert report_types == [ReportTypes.PERFORMANCE]
        assert automl_stats is None

    def test_passed_report_types_are_not_overwritten(
        self, measurements_path: Path
    ):
        """
        Tests that explicitly provided report types are used as they are.
        """
        _, report_types, _ = Report.load_measurements_for_report(
            [str(measurements_path)],
            skip_unoptimized_model=True,
            report_types=[ReportTypes.CLASSIFICATION],
        )

        assert report_types == [ReportTypes.CLASSIFICATION]

    def test_no_measurements_files(self):
        """
        Tests that an empty list of measurements files is reported as
        a missing report type.
        """
        with pytest.raises(ArgumentError, match="cannot be deduced"):
            Report.load_measurements_for_report(
                [], skip_unoptimized_model=True
            )
