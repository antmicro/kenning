#!/usr/bin/env python3
# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Checks whether Kenning's extras can be installed on a given Python version.

Usage:

    python tools/check_python_compatibility.py 3.14
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from pyproject_requirements import Pyproject

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def uv_resolve(
    requirements: List[str], version: str, root: Path
) -> Optional[str]:
    """
    Resolves requirement list for a Python version.

    Parameters
    ----------
    requirements : List[str]
        Requirements
    version : str
        Python version
    root : Path
        Project directory

    Returns
    -------
    Optional[str]
        None if requirements resolve, uv's error output otherwise
    """
    command = [
        "uv", "pip", "compile", "-",
        "--python-version", version,
        "--quiet", "--no-header", "--no-annotate",
    ]
    process = subprocess.run(
        command,
        input="\n".join(requirements),
        capture_output=True,
        text=True,
        cwd=root,
    )
    if process.returncode == 0:
        return None
    return process.stderr.strip()


def render(version: str, errors: Dict[str, Optional[str]]) -> str:
    """
    Renders results as Markdown.

    Parameters
    ----------
    version : str
        Checked Python version
    errors : Dict[str, Optional[str]]
        uv's error for each extra, None if the extra resolves

    Returns
    -------
    str
        Markdown report
    """
    lines = [
        f"# Kenning extras on Python {version} ({PLATFORM})",
        "",
        "| extra | status |",
        "|---|---|",
    ]
    for extra, error in errors.items():
        lines.append(f"| {extra} | {'ok' if error is None else 'failed'} |")

    for extra, error in errors.items():
        if error is not None:
            lines += ["", f"## {extra}", "", "```", error, "```"]
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    """
    Checks extras on a Python version and prints the report.

    Parameters
    ----------
    argv : Optional[List[str]]
        Command-line arguments, sys.argv is used if None

    Returns
    -------
    int
        1 if any extra cannot be installed, 0 otherwise
    """
    parser = argparse.ArgumentParser(
        description="Checks whether Kenning's extras can be installed on a "
        "given Python version."
    )
    parser.add_argument("python", help="Python version, e.g. 3.14")
    parser.add_argument("--pyproject", type=Path, default=PYPROJECT)
    args = parser.parse_args(argv)

    pyproject = Pyproject.load(args.pyproject)
    extras = pyproject.get_extras()
    root = args.pyproject.resolve().parent

    def resolve(requirements: List[str]) -> Optional[str]:
        return uv_resolve(requirements, args.python, root)

    with ThreadPoolExecutor() as pool:
        errors = dict(zip(extras, pool.map(resolve, extras.values())))

    print(render(args.python, errors))
    return 1 if any(error is not None for error in errors.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
