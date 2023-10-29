# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing functionalities for Kenning dependencies.
"""

import sys
from typing import Dict, List

if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path
if sys.version_info.minor < 11:
    import tomli as tomllib
else:
    import tomllib

from kenning.resources.pyproject import PYPROJECT_FILE


DEPENDENCIES: Dict[str, List[str]] = None


def get_all_dependencies() -> Dict[str, List[str]]:
    """
    Gets Kenning dependencies.

    It buffers them, to avoid reading file multiple times.

    Returns
    -------
    Dict[str, List[str]]
        Optional and normal dependencies from pyproject.toml
    """
    global DEPENDENCIES
    if DEPENDENCIES is None:
        DEPENDENCIES = _parse_dependencies()
    return DEPENDENCIES


def _parse_dependencies() -> Dict[str, List[str]]:
    """
    Reads pyproject.toml and returns dependencies.

    Returns
    -------
    Dict[str, List[str]]
        Optional and normal dependencies from pyproject.toml
    """
    with path(__package__, PYPROJECT_FILE) as dependencies:
        with open(dependencies, "rb") as fd:
            pyproject = tomllib.load(fd)
    return pyproject["project"]["optional-dependencies"] | {
        "kenning": pyproject["project"]["dependencies"]
    }
