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
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set

from packaging.requirements import Requirement
from pyproject_requirements import Pyproject

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

OK = "✓"
FAILED = "✗"
UNTESTED = "?"


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
        "uv",
        "pip",
        "compile",
        "-",
        "--python-version",
        version,
        "--quiet",
        "--no-header",
        "--no-annotate",
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


def unsupported(requirements: List[str], version: str) -> List[str]:
    """
    Lists packages without a requirement line for a Python version.

    Parameters
    ----------
    requirements : List[str]
        Requirements
    version : str
        Python version, e.g. 3.14

    Returns
    -------
    List[str]
        Names of packages none of whose lines apply to the version
    """
    environment = {
        "python_version": version,
        "python_full_version": f"{version}.0",
    }
    applies: Dict[str, bool] = {}
    for line in requirements:
        requirement = Requirement(line)
        applies[requirement.name] = applies.get(requirement.name, False) or (
            requirement.marker is None
            or requirement.marker.evaluate(environment)
        )
    return sorted(name for name, ok in applies.items() if not ok)


def is_untested(
    requirements: List[str], base: List[str], version: str
) -> bool:
    """
    Checks whether no package of an extra applies to a Python version,
    so resolving it only checks the base requirements.

    Parameters
    ----------
    requirements : List[str]
        Requirements of the extra, including the base ones
    base : List[str]
        Base requirements of the project
    version : str
        Python version, e.g. 3.14

    Returns
    -------
    bool
        True if all packages the extra adds to the base are dropped
    """
    own = {Requirement(line).name for line in requirements if line not in base}
    return bool(own) and own <= set(unsupported(requirements, version))


def render(
    version: str,
    errors: Dict[str, Optional[str]],
    untested: Set[str],
    show_errors: bool = False,
) -> str:
    """
    Renders results as Markdown.

    Parameters
    ----------
    version : str
        Checked Python version
    errors : Dict[str, Optional[str]]
        uv's error for each extra, None if the extra resolves
    untested : Set[str]
        Extras whose own packages were all dropped, so only the base
        requirements were resolved
    show_errors : bool
        Whether to append uv's error output of failed extras

    Returns
    -------
    str
        Markdown report
    """
    statuses = {}
    for extra, error in errors.items():
        if error is not None:
            statuses[extra] = FAILED
        elif extra in untested:
            statuses[extra] = UNTESTED
        else:
            statuses[extra] = OK

    width = max(len("extra"), *map(len, statuses))
    lines = [
        f"# Kenning extras on Python {version} "
        f"({platform.system()} {platform.machine()})",
        "",
        f"| {'extra':<{width}} | status |",
        f"|{'-' * (width + 2)}|--------|",
    ]
    for extra, status in statuses.items():
        lines.append(f"| {extra:<{width}} | {status:^6} |")

    counts = [
        f"{list(statuses.values()).count(status)} {status}"
        for status in (OK, FAILED, UNTESTED)
    ]
    lines += ["", ", ".join(counts)]

    failed = [extra for extra, error in errors.items() if error is not None]
    if show_errors:
        for extra in failed:
            lines += ["", f"## {extra}", "", "```", errors[extra], "```"]
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
        1 if any extra cannot be installed or was not tested,
        0 otherwise
    """
    parser = argparse.ArgumentParser(
        description="Checks whether Kenning's extras can be installed on a "
        "given Python version."
    )
    parser.add_argument("python", help="Python version, e.g. 3.14")
    parser.add_argument("--pyproject", type=Path, default=PYPROJECT)
    parser.add_argument(
        "--errors",
        action="store_true",
        help="Print uv's error output for failed extras",
    )
    args = parser.parse_args(argv)

    pyproject = Pyproject.load(args.pyproject)
    extras = pyproject.get_extras()
    root = args.pyproject.resolve().parent

    def resolve(requirements: List[str]) -> Optional[str]:
        return uv_resolve(requirements, args.python, root)

    with ThreadPoolExecutor() as pool:
        errors = dict(zip(extras, pool.map(resolve, extras.values())))

    base = extras[pyproject.name]
    untested = {
        extra
        for extra, requirements in extras.items()
        if is_untested(requirements, base, args.python)
    }

    print(render(args.python, errors, untested, args.errors))
    failed = any(error is not None for error in errors.values())
    return 1 if failed or untested else 0


if __name__ == "__main__":
    sys.exit(main())
