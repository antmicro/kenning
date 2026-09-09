# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

import pytest

from kenning.report.markdown_report import MarkdownReport


def measurements_data(model_name: str, accuracy_shift: int = 0) -> dict:
    """
    Returns measurements covering the classification and performance
    report types.
    """
    return {
        "model_name": model_name,
        "eval_confusion_matrix": [
            [8 + accuracy_shift, 2 - accuracy_shift],
            [1, 9],
        ],
        "class_names": ["first", "second"],
        "target_inference_step": [0.1, 0.2, 0.3],
        "target_inference_step_timestamp": [0.0, 0.1, 0.2],
        "session_utilization_timestamp": [0.0, 0.1, 0.2],
        "session_utilization_mem_percent": [10.0, 20.0, 30.0],
        "session_utilization_cpus_percent": [[10.0], [20.0], [30.0]],
        "compiled_model_size": 1000,
    }


@pytest.fixture
def measurements_file(tmp_path: Path) -> Path:
    """
    Provides a single measurements file.
    """
    path = tmp_path / "measurements.json"
    path.write_text(json.dumps(measurements_data("model")))
    return path


@pytest.fixture
def measurements_files(tmp_path: Path) -> list:
    """
    Provides measurements files of two models for comparison reports.
    """
    paths = []
    for i, name in enumerate(("first", "second")):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(measurements_data(name, i)))
        paths.append(path)
    return paths


class TestMarkdownReportArguments:
    def test_measurements_are_required(self, tmp_path: Path):
        """
        Tests that a report without measurements and without the to-html
        flag is rejected.
        """
        with pytest.raises(argparse.ArgumentError, match="--measurements"):
            MarkdownReport(report_path=tmp_path / "report.md")

    def test_html_requires_existing_report(self, tmp_path: Path):
        """
        Tests that rendering HTML of a non-existing report is rejected.
        """
        with pytest.raises(argparse.ArgumentError, match="--report-path"):
            MarkdownReport(report_path=tmp_path / "report.md", to_html=True)

    def test_html_of_existing_report_needs_no_measurements(
        self, tmp_path: Path
    ):
        """
        Tests that an existing report can be rendered to HTML without
        providing any measurements.
        """
        report_path = tmp_path / "report.md"
        report_path.touch()

        report = MarkdownReport(report_path=report_path, to_html=True)

        assert report.to_html == tmp_path / "report"

    def test_image_formats_include_html_by_default(
        self, measurements_file: Path, tmp_path: Path
    ):
        """
        Tests that interactive plots are generated unless disabled.
        """
        report = MarkdownReport(
            measurements=[measurements_file],
            report_path=tmp_path / "report.md",
        )

        assert report.image_formats == {"png", "html"}

    def test_only_png_images(self, measurements_file: Path, tmp_path: Path):
        """
        Tests that only static plots are generated when requested.
        """
        report = MarkdownReport(
            measurements=[measurements_file],
            report_path=tmp_path / "report.md",
            only_png_images=True,
        )

        assert report.image_formats == {"png"}

    def test_comparison_only_is_disabled_for_single_model(
        self, measurements_file: Path, tmp_path: Path
    ):
        """
        Tests that a comparison of a single model falls back
        to a standard report.
        """
        report = MarkdownReport(
            measurements=[measurements_file],
            report_path=tmp_path / "report.md",
            comparison_only=True,
        )

        assert report.comparison_only is False

    def test_mismatched_model_names_are_ignored(
        self, measurements_file: Path, tmp_path: Path
    ):
        """
        Tests that model names are dropped when their number
        does not match the number of measurements.
        """
        report = MarkdownReport(
            measurements=[measurements_file],
            report_path=tmp_path / "report.md",
            model_names=["first", "second"],
        )

        assert report.model_names is None


class TestMarkdownReportGeneration:
    def test_report_is_generated_for_single_model(
        self, measurements_file: Path, tmp_path: Path
    ):
        """
        Tests that a report along with its plots is generated for
        the deduced report types.
        """
        report_path = tmp_path / "report.md"
        report = MarkdownReport(
            measurements=[measurements_file],
            report_path=report_path,
            only_png_images=True,
        )

        report.generate_report(subcommands=[], command=["kenning", "report"])

        content = report_path.read_text()
        assert "Confusion matrix" in content
        assert "Inference time" in content
        assert (tmp_path / "img" / "confusion_matrix.png").exists()
        assert (tmp_path / "img" / "inference_time.png").exists()

    def test_report_name_is_deduced(
        self, measurements_file: Path, tmp_path: Path
    ):
        """
        Tests that the report name is deduced from measurements
        when it is not given.
        """
        report = MarkdownReport(
            measurements=[measurements_file],
            report_path=tmp_path / "report.md",
            report_types=["classification"],
            only_png_images=True,
        )

        report.generate_report(subcommands=[], command=[])

        assert "model" in report.report_name

    def test_given_report_name_is_kept(
        self, measurements_file: Path, tmp_path: Path
    ):
        """
        Tests that an explicitly given report name is not overwritten.
        """
        report_path = tmp_path / "report.md"
        report = MarkdownReport(
            measurements=[measurements_file],
            report_path=report_path,
            report_name="Custom report",
            report_types=["classification"],
            only_png_images=True,
        )

        report.generate_report(subcommands=[], command=[])

        assert report.report_name == "Custom report"
        assert "Custom report" in report_path.read_text()

    def test_comparison_report_is_generated(
        self, measurements_files: list, tmp_path: Path
    ):
        """
        Tests that a report of several models contains comparison sections
        and per model sections with prefixed plots.
        """
        report_path = tmp_path / "report.md"
        report = MarkdownReport(
            measurements=measurements_files,
            report_path=report_path,
            report_types=["classification", "performance"],
            only_png_images=True,
        )

        report.generate_report(subcommands=[], command=[])

        imgdir = tmp_path / "img"
        assert (imgdir / "classification_metric_comparison.png").exists()
        assert (imgdir / "first_confusion_matrix.png").exists()
        assert (imgdir / "second_confusion_matrix.png").exists()
        assert "Comparison" in report_path.read_text()

    def test_comparison_only_report_skips_model_sections(
        self, measurements_files: list, tmp_path: Path
    ):
        """
        Tests that per model sections are not included in a report limited
        to comparisons.
        """
        report_path = tmp_path / "report.md"
        report = MarkdownReport(
            measurements=measurements_files,
            report_path=report_path,
            report_types=["classification"],
            only_png_images=True,
            comparison_only=True,
        )

        report.generate_report(subcommands=[], command=[])

        content = report_path.read_text()
        assert "Comparison" in content
        assert "Confusion matrix" not in content

    def test_model_names_override_the_ones_from_measurements(
        self, measurements_files: list, tmp_path: Path
    ):
        """
        Tests that model names given by the user are used in the report.
        """
        report_path = tmp_path / "report.md"
        report = MarkdownReport(
            measurements=measurements_files,
            report_path=report_path,
            report_types=["classification"],
            model_names=["renamed_first", "renamed_second"],
            only_png_images=True,
        )

        report.generate_report(subcommands=[], command=[])

        content = report_path.read_text()
        assert "renamed_first" in content
        assert "renamed_second" in content

    def test_comparison_only_of_single_model_is_rejected(
        self, measurements_file: Path, tmp_path: Path
    ):
        """
        Tests that a comparison of a single model is rejected.
        """
        report = MarkdownReport(
            measurements=[measurements_file],
            report_path=tmp_path / "report.md",
            report_types=["classification"],
            only_png_images=True,
        )
        report.comparison_only = True

        with pytest.raises(argparse.ArgumentError, match="--comparison-only"):
            report.generate_report(subcommands=[], command=[])

    def test_html_report_is_rendered(
        self, measurements_file: Path, tmp_path: Path, mocker
    ):
        """
        Tests that the HTML rendering is requested for the generated
        report.
        """
        generate_html_report = mocker.patch(
            "kenning.report.markdown_report.generate_html_report"
        )
        report_path = tmp_path / "report.md"
        report = MarkdownReport(
            measurements=[measurements_file],
            report_path=report_path,
            report_types=["classification"],
            only_png_images=True,
            to_html=True,
        )

        report.generate_report(subcommands=[], command=[])

        generate_html_report.assert_called_once()
        assert generate_html_report.call_args.args[0] == report_path
        assert generate_html_report.call_args.args[1] == tmp_path / "report"
