# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used to generate Zephelin Traces report segment, containing a link to
Zephelin Trace Viewer with traca data baked in.
"""

import base64
import tempfile
from importlib.resources import path
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

from kenning.core.exceptions import KenningReportError
from kenning.report.markdown_components.general import (
    create_report_from_measurements,
)
from kenning.resources import reports
from kenning.utils.resource_manager import ResourceManager

ZEPHYR_TRACE_VIEWER_URL = (
    "https://antmicro.github.io/zephelin-trace-viewer/single-html/index.html"
)


def zephyr_traces_report(
    measurementsdata: Dict[str, Any],
    imgdir: Path,
    imgprefix: str,
    root_dir: Path,
    image_formats: Set[str],
    measurements_path: Path,
    compiled_model_path: Optional[Path] = None,
    zephyr_trace_file_ctf: Optional[Path] = None,
    zephyr_trace_file_tef: Optional[Path] = None,
    zephyr_build_path: Optional[Path] = None,
    zephyr_base: Optional[Path] = None,
    cfg: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[str, Dict]:
    """
    Creates zephyr traces section of the report.

    Parameters
    ----------
    measurementsdata : Dict[str, Any]
        Not used in this function.
    imgdir : Path
        Not used in this function.
    imgprefix : str
        Not used in this function.
    root_dir : Path
        Path to the root of the documentation project involving this report.
    image_formats : Set[str]
        Not used in this function.
    measurements_path: Path
        Path to the measurements JSON file used to generate the report (used to
        retrieve the traces).
    compiled_model_path: Optional[Path]
        Path to the model used to generate the traces (used for CTF->TEF
        conversion)
    zephyr_trace_file_ctf: Optional[Path]
        Path to a trace file from Zephyr in CTF format (will be converted to
        TEF)
    zephyr_trace_file_tef: Optional[Path]
        Path to a trace file from zephyr in TEF format.
    zephyr_build_path: Optional[Path]
        Path to zephyr build directory with the traced program (used for
        CTF->TEF conversion)
    zephyr_base: Optional[Path]
        Path to Zephyr repository (used for CTF->TEF conversion).
    cfg: Optional[str]
        Name of the config file used to generate the report. Will be used to
        infer the HTML page title for Zephyr Traces Report. Set to None if no
        config file was used.
    **kwargs : Any
        Additional keyword arguments (not used here).

    Returns
    -------
    Tuple[str, Dict]
        Content of the report in MyST format, a dict of measurements.

    Raises
    ------
    KenningReportError
        All methods of retrieving Zephyr traces failed.
    """
    from kenning.platforms.zephyr import _prepare_traces
    from kenning.utils.zpl_suffix import ZplSuffix

    # We extract Zephyr traces from available sources.
    #
    # Trace source priority:
    #  1. If `zephyr_trace_file_tef` was provided, we use it.
    #  2. If 'zephyr_trace_file_ctf' was provided, we convert it to TEF and use
    #     it (`compiled_model_path` is necessary for the conversion,
    #     'zephyr_base' and `zephyr_build_path` may be provided).
    #  3. If a file exists at the default TEF file path (inferred from
    #     `measurements_path` - this is were traces collected and converted by
    #     Kenning itself are stored), we use it.
    #  4. If a file exists at the default CTF file path (inferred from
    #     `measurements_path` - this is were traces collected by Kenning itself
    #     are stored), we use it.

    path_tef = ZplSuffix.TRACE_JSON._get_path_with_suffix(measurements_path)
    convert_needed = False
    path_ctf = None
    if zephyr_trace_file_tef is not None:
        path_tef = zephyr_trace_file_tef
    elif zephyr_trace_file_ctf is not None:
        path_ctf = zephyr_trace_file_ctf
        convert_needed = True
    else:
        if not path_tef.is_file():
            convert_needed = True
            path_ctf = ZplSuffix.CTF._get_path_with_suffix(measurements_path)
            if not path_ctf.is_file():
                raise KenningReportError(
                    "Zephyr trace file path was not provided and the file is"
                    " not in the default location. Please provide a file path"
                    " using --zephyr-trace-file-ctf (for a CTF file) or"
                    " --zephyr-trace-file-tef for a TEF JSON file."
                )

    if convert_needed:
        optimizer = measurementsdata["optimizers"][-1]["compiler_framework"]
        from kenning.optimizers.tflite import TFLiteCompiler
        from kenning.optimizers.tvm import TVMCompiler

        _prepare_traces(
            TVMCompiler
            if optimizer == "tvm"
            else (TFLiteCompiler if optimizer == "tflite" else None),
            path_ctf,
            path_tef,
            zephyr_build_path,
            zephyr_base,
        )

    # We download Zephelin Trace Viewer with JS and CSS inlined into a single
    # HTML file (that's because otherwise browser CORS policy won't allow to
    # run the app locally without a server).

    resource_manager = ResourceManager()
    ztv_path = Path(tempfile.gettempdir()) / "zephelin_trace_viewer.html"
    resource_manager.get_resource(ZEPHYR_TRACE_VIEWER_URL, ztv_path)

    # Browser safety policy won't allow the Zephelin Trace Viewer app to
    # open a file (with traces) by itself, so to create a link to the app with
    # traces opened automatically we need to bake trace data into the HTML.
    # The app automatically detects if window.initialTraces tag is present
    # and if so, automatically parses and uses them.
    # Therefore all we need to do is paste a <script> into the HTML, that
    # creates the tag and updsbase64-encoded trace data into it.

    with open(ztv_path) as ztv:
        with open(path_tef) as tracefile:
            from bs4 import BeautifulSoup as Soup

            # Baking trace data into the HTML.
            raw_file = tracefile.read()
            traces = raw_file.encode("ascii")
            traces_encoded = base64.b64encode(traces).decode("ascii")
            code = ztv.read()
            dom_tree = Soup(code)
            dom_tree.head.append(
                Soup(
                    f"<script>window.initialTraces='{traces_encoded}';</script>"
                )
            )
            # Changing page title to match Kenning Report.
            dom_tree.head.title.replace_with(
                Soup(
                    f"<title>{cfg} - Kenning Report for Zephyr Traces</title>"
                    if cfg
                    else "<title>Kenning Report for Zephyr Traces</title>"
                ),
            )
            # Generating the finished HTML file containing Zehpelin Trace
            # Viewer with traces baked-in.
            with open(
                root_dir / "zephyr_traces_report.html", "w"
            ) as baked_ztv:
                baked_ztv.write(dom_tree.prettify())

    # Generating a section of the report with link to the created Zephelin
    # Trace Viewer HTML

    with path(reports, "zephyr_traces.md") as reporttemplate:
        return create_report_from_measurements(reporttemplate, {}), None
