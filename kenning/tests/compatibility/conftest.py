import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Type

import pandas as pd
import pytest
from _pytest.terminal import TerminalReporter

from kenning.utils.class_loader import (
    get_all_subclasses,
    get_base_classes_dict,
)

COMPATIBILITY_PYTEST_MARKER = "compat_matrix"
COMPATIBILITY_MARKER = "COMPAT"
DELIM = ":"


@pytest.hookimpl
def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: List[pytest.Item]
):
    cls_to_path = {cls: path for path, cls in get_base_classes_dict().values()}

    for item in items:
        if mark := item.get_closest_marker(COMPATIBILITY_PYTEST_MARKER):
            # Retrieve blocks
            cls1, cls2 = mark.args
            path1, path2 = cls_to_path[cls1], cls_to_path[cls2]

            # Add metadata
            item.add_marker(DELIM.join([COMPATIBILITY_MARKER, path1, path2]))


@pytest.hookimpl
def pytest_terminal_summary(
    terminalreporter: TerminalReporter,
    exitstatus: pytest.ExitCode,
    config: pytest.Config,
):
    path_to_cls = {path: cls for path, cls in get_base_classes_dict().values()}
    status_mapping = {
        "passed": "Compatible",
        "xpassed": "Unexpected Compatible",
        "failed": "Unexpected Incompatible",
        "xfailed": "Incompatible",
        "error": "Error",
        "skipped": "—",
    }
    status_mapping = defaultdict(lambda: "Unknown", status_mapping)
    dataframes = {}

    # Collect results into dataframes
    for outcome in status_mapping.keys():
        status = status_mapping[outcome]
        for report in terminalreporter.stats.get(outcome, []):
            if mark := get_compatibility_mark(report.keywords):
                report_to_entry(report, mark, status, dataframes, path_to_cls)

    # Save dataframes
    for (base1, base2), df in dataframes.items():
        name = Path(config.option.test_compat_dir)
        name.mkdir(exist_ok=True, parents=True)
        name /= f"{base1.__name__}-{base2.__name__}.csv"
        df.to_csv(name)


def get_compatibility_mark(marks: Iterable[str]) -> Optional[str]:
    for mark in marks:
        if COMPATIBILITY_MARKER in mark:
            return mark


def report_to_entry(
    report: pytest.TestReport,
    mark: str,
    status: str,
    dataframes: Dict[Tuple[Type, Type], pd.DataFrame],
    path_to_cls: Dict[str, Type],
):
    # Extract concrete classes
    _, path1, path2 = mark.split(DELIM)
    base1, base2 = path_to_cls[path1], path_to_cls[path2]
    _, _, test_name = report.location
    match = re.search(r"(?<=\[).*(?=\])", test_name)
    assert match
    concrete1, concrete2 = match.group().split("-")

    # Create empty dataframe if it doesn't exist
    if (base1, base2) not in dataframes:
        subclasses1 = get_all_subclasses(path1, base1)
        subclasses2 = get_all_subclasses(path2, base2)
        data = {cls.__name__: [None] * len(subclasses1) for cls in subclasses2}
        dataframes[(base1, base2)] = pd.DataFrame(
            data=data, index=[cls.__name__ for cls in subclasses1]
        )

    dataframes[(base1, base2)].loc[concrete1, concrete2] = status


def pytest_addoption(parser: pytest.Parser):
    """
    Adds argparse options to parser.
    """
    parser.addoption(
        "--test-compat-dir",
        action="store",
        default="./compatibility",
        help="Directory used to store compatibility matrices",
    )
