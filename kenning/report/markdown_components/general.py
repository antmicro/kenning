# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with general function used by different modules and functions in
markdown_components package.
"""

from collections import namedtuple
from importlib.resources import path
from pathlib import Path
from typing import Dict, List, Optional

from kenning.core.metrics import Metric
from kenning.resources import reports
from kenning.utils.logger import KLogger


def get_plot_wildcard_path(plot_path: Path, root_dir: Path) -> str:
    """
    Generate wildcard plot path relative to given directory
    which can be used in report.

    Parameters
    ----------
    plot_path : Path
        Path to the saved plot.
    root_dir : Path
        Report root directory.

    Returns
    -------
    str
        Universal plot path relative to report root directory.
    """
    return str(
        plot_path.with_suffix(plot_path.suffix + ".*").relative_to(root_dir)
    )


def create_report_from_measurements(
    template: Path, measurementsdata: Dict[str, List]
) -> str:
    """
    Creates a report from template and measurements data.

    It additionally provides methods for quality metrics.

    Parameters
    ----------
    template : Path
        Path to the Jinja template.
    measurementsdata : Dict[str, List]
        Dictionary describing measurements taken during benchmark.

    Returns
    -------
    str
        Content of the report.
    """
    with open(template, "r") as resourcetemplatefile:
        from jinja2 import Template

        resourcetemplate = resourcetemplatefile.read()
        tm = Template(resourcetemplate)

        content = tm.render(data=measurementsdata, zip=zip, Metric=Metric)

        return content


def generate_html_report(
    report_path: Path,
    output_folder: Path,
    debug: bool = False,
    override_conf: Optional[Dict] = None,
):
    """
    Runs Sphinx with HTML builder for generated report.

    Parameters
    ----------
    report_path : Path
        Path to the generated report file
    output_folder : Path
        Where generated HTML report should be saved
    debug : bool
        Debug mode -- allows to print more information
    override_conf : Optional[Dict]
        Custom configuration of Sphinx app
    """
    from sphinx.application import Sphinx
    from sphinx.cmd.build import handle_exception
    from sphinx.util.docutils import docutils_namespace, patch_docutils

    with path(reports, "conf.py") as _conf:
        override_conf = (override_conf or {}) | {
            # Include only report file
            "include_patterns": [f"{report_path.name}"],
            # Ensure report file isn't excluded
            "exclude_patterns": [],
            # Use report file as main source
            "master_doc": f'{report_path.with_suffix("").name}',
            # Static files for HTML
            "html_static_path": [f'{_conf.parent / "_static"}'],
            # Remove PFD button
            "html_theme_options.pdf_url": [],
            # Warning about using h2 header
            "suppress_warnings": ["myst.header"],
        }
        app = None
        try:
            with patch_docutils(_conf.parent), docutils_namespace():
                app = Sphinx(
                    report_path.parent,
                    _conf.parent,
                    output_folder,
                    output_folder / ".doctrees",
                    "html",
                    override_conf,
                    freshenv=False,
                )
                app.build(False, [str(report_path)])
        except Exception as ex:
            mock_args = namedtuple(
                "MockArgs", ("pdb", "verbosity", "traceback")
            )(pdb=debug, verbosity=debug, traceback=debug)
            handle_exception(app, mock_args, ex)
            KLogger.error(
                "Error occurred, HTML report won't be generated",
                ex.args,
                stack_info=True,
            )
