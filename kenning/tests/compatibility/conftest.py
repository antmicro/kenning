import re
import shutil
from collections import defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Type

import pandas as pd
import pytest
from _pytest.terminal import TerminalReporter
from bokeh.models import (
    ColumnDataSource,
    DataTable,
    HTMLTemplateFormatter,
    TableColumn,
)
from jinja2 import Template

from kenning.core.drawing import DEFAULT_PLOT_SIZE, Plot, choose_theme
from kenning.resources import reports
from kenning.scenarios.render_report import (
    generate_html_report,
    get_plot_wildcard_path,
)
from kenning.utils.class_loader import (
    get_all_subclasses,
    get_base_classes_dict,
)

ANSI_COLOR = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
COMPATIBILITY_PYTEST_MARKER = "compat_matrix"
COMPATIBILITY_MARKER = "COMPAT"
DELIM = ":"
STATUS_MAPPING = {
    "passed": "Compatible",
    "xpassed": "Fixed",
    "failed": "Failed",
    "xfailed": "Incompatible",
    "error": "Error",
    "skipped": "—",
}
STATUS_MAPPING = defaultdict(lambda: "Unknown", STATUS_MAPPING)


def pair_to_str(cls1: Type, cls2: Type) -> str:
    return f"{cls1.__name__}-{cls2.__name__}"


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
    dataframes = {}
    logs = {}
    stats = defaultdict(
        # Do not include "skipped" into displayed statistics
        lambda: {
            stat: 0
            for outcome, stat in STATUS_MAPPING.items()
            if outcome != "skipped"
        }
    )

    # Collect results into dataframes
    for outcome, status in STATUS_MAPPING.items():
        for report in terminalreporter.stats.get(outcome, []):
            if mark := get_compatibility_mark(report.keywords):
                report_to_entry(
                    report=report,
                    mark=mark,
                    status=status,
                    dataframes=dataframes,
                    stats=stats,
                    path_to_cls=path_to_cls,
                    logs=logs,
                )

    generate_compatibility_report(stats, dataframes, logs, config)


def get_compatibility_mark(marks: Iterable[str]) -> Optional[str]:
    for mark in marks:
        if COMPATIBILITY_MARKER in mark:
            return mark


def report_to_entry(
    report: pytest.TestReport,
    mark: str,
    status: str,
    dataframes: Dict[Tuple[Type, Type], pd.DataFrame],
    stats: Dict,
    path_to_cls: Dict[str, Type],
    logs: Dict[str, Dict],
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

    # Save data
    pair_stats = stats[pair_to_str(base1, base2)]
    if status in pair_stats:
        pair_stats[status] += 1
    dataframes[(base1, base2)].loc[concrete1, concrete2] = status
    logs[f"{concrete1}-{concrete2}"] = get_report_log(report)


def get_report_log(report: pytest.TestReport):
    sections = report.sections[:]
    if report.longreprtext.strip():
        sections.append(("Test result", report.longreprtext))

    new_sections = {}
    for caption, content in sections:
        lines = []
        for line in content.split("\n"):
            line = ANSI_COLOR.sub("", line)
            line = line.replace("\r", "")
            line = line.replace("\x08", "")
            lines.append(line)
        content = "\n".join(lines)
        new_sections[caption] = content

    return new_sections


class CompatibilityPlot(Plot):
    def __init__(
        self,
        data: pd.DataFrame,
        title: Optional[str] = None,
        width: int = DEFAULT_PLOT_SIZE,
        height: int = DEFAULT_PLOT_SIZE // 3,
    ):
        super().__init__(width, height, title)
        self.data = data

    def plot_bokeh(
        self,
        output_path: Optional[Path],
        output_formats: Iterable[str],
    ):
        self.data = self.data.reset_index().rename({"index": "Block"}, axis=1)
        source = ColumnDataSource(self.data)
        columns = [
            TableColumn(field=col, title=col, formatter=self.formatter(col))
            for col in self.data.columns
        ]
        data_table = DataTable(
            source=source,
            columns=columns,
            height=DataTable.row_height.property._default
            * (self.data.shape[0] + 1),
            autosize_mode="fit_viewport",
            sortable=False,
            selectable=False,
        )
        self._output_bokeh_figure(
            data_table,
            output_path,
            output_formats,
        )

    def plot_matplotlib(
        self,
        output_path: Optional[Path],
        output_formats: Iterable[str],
    ):
        pass

    @staticmethod
    def formatter(column_name: str):
        return HTMLTemplateFormatter(
            template=f"""
            <%=(function formatter(){{
                const cls1 = Block;
                const cls2 = "{column_name}";
                const supported = [
                    'Compatible',
                    'Incompatible',
                    'Fixed',
                    'Failed',
                ];
                if (supported.includes(value))
                    return `
                        <a
                          class="${{value.toLowerCase()}}"
                          href="logs/${{cls1}}-${{cls2}}.html"
                        >${{value}}</a>
                    `;
                return `
                    <div
                      style="color: var(--md-default-fg-color)"
                    >${{value}}</div>
                `;
            }}()) %>"""
        )


def generate_compatibility_report(
    stats: Dict,
    dataframes: Dict[Tuple[Type, Type], pd.DataFrame],
    logs: Dict[str, Dict],
    config: pytest.Config,
):
    root = Path(config.option.test_compat_dir)
    if root.exists():
        shutil.rmtree(root)
    report_path = root / "index.md"
    html_path = root / "html"
    logdir = root / "logs"
    imgdir = root / "img"
    logdir.mkdir(exist_ok=True, parents=True)
    imgdir.mkdir(exist_ok=True, parents=True)

    # Save logs
    for log, sections in logs.items():
        content = "\n".join(
            f"{'-' * 20} {section_name} {'-' * 20}\n{content}"
            for section_name, content in sections.items()
        )
        (logdir / f"{log}.log").write_text(content)
        content = content.replace("<", "&lt;").replace(">", "&gt;")
        (logdir / f"{log}.html").write_text(f"<pre>{content}</pre>")

    # Create plots and collect report data
    data = {
        "names": [],
        # name -> path
        "paths": {},
        # name -> stats
        "stats": dict(stats),
    }
    with choose_theme(custom_bokeh_theme=True, custom_matplotlib_theme=False):
        for (base1, base2), df in dataframes.items():
            name = pair_to_str(base1, base2)
            plot_path = imgdir / name
            data["names"].append(name)
            data["paths"][name] = get_plot_wildcard_path(plot_path, root)
            CompatibilityPlot(df, title=name).plot(
                plot_path,
                output_formats=["html"],
                backend="bokeh",
            )

    # Save Markdown report
    templatepath = files(reports).joinpath("compatibility.md")
    resourcetemplate = templatepath.read_text()
    tm = Template(resourcetemplate)
    content = tm.render(data=data, zip=zip)
    report_path.write_text(content)

    # Save html report
    generate_html_report(
        report_path,
        html_path,
        override_conf={
            "html_css_files": ["css/bokeh.css", "css/compatibility.css"]
        },
    )

    # Move html report to the root
    for dir_or_file in html_path.iterdir():
        shutil.move(dir_or_file, root)


def pytest_addoption(parser: pytest.Parser):
    """
    Adds argparse options to parser.
    """
    parser.addoption(
        "--test-compat-dir",
        action="store",
        default="./docs/source/generated/compatibility",
        help="Directory used to store compatibility matrices",
    )
