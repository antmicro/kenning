# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A package that provides Report-related class used mainly to retrieve
basic report parameters.
"""


from pathlib import Path
from typing import List, Optional

from kenning.core.report import Report


class StubReport(Report):
    """
    A dummy Report derive class used to retrieve basic Report related
    parameters, used mainly in InferenceTester when report
    generation is not required.
    """

    arguments_structure = {}

    def __init__(
        self,
        measurements: Path | List[Path] = [None],
        report_name: Optional[str] = None,
        report_types: Optional[List[str]] = None,
        automl_stats: Optional[Path] = None,
    ):
        super().__init__(measurements, report_name, report_types, automl_stats)

    def generate_report(
        self,
        subcommands: Optional[List[str]] = None,
        command: Optional[List[str]] = None,
        automl_stats: Optional[Path] = None,
    ) -> None:
        return None
