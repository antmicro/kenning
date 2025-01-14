# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import re
from glob import glob
from pathlib import Path
from typing import Set

import pytest
from isort import stdlibs

from kenning.utils.excepthook import find_missing_optional_dependency

RE_IMPORT_NAME = r"^\s*(import ([^ \.]+))|(from ([^ \.]+)(\.[^ \.]+)* import)"
# Ignored modules (don't count into the threshold) extended with stdlibs
IGNORED_MODULES = {
    # ROS dependencies -- manual installation
    "sensor_msgs",
    "kenning_computer_vision_msgs",
    "rclpy",
    "tflite_runtime",  # fallback to tensorflow
    "__future__",  # builtin module not in stdlib
    # sparse kernel dependencies - installed separately
    "cutlass_library",
    "setuptools",
} | stdlibs.py3.stdlib

KNOWN_MAPS = {
    "jsonrpc": "json-rpc",
    "sklearn": "scikit_learn",
}


def get_all_used_imports() -> Set[str]:
    """
    Reads all Python files in project and searches
    for names of imported modules.

    Returns
    -------
    Set[str]
        Modules imported across whole project
    """
    import kenning

    imports = set()
    _re = re.compile(RE_IMPORT_NAME)
    for py_file in glob(str(Path(kenning.__file__).parent / "**" / "*.py")):
        with open(py_file, "r") as fd:
            for line in fd:
                match = _re.match(line.removesuffix("\n"))
                if match:
                    imports.add(
                        match.group(2) if match.group(2) else match.group(4)
                    )
    imports.remove("kenning")
    return imports


@pytest.mark.fast
class TestFindingDependency:
    def test_find_missing_dependency(self):
        missing = []
        non_stdlib_modules = 0
        for missing_module in get_all_used_imports():
            if missing_module in IGNORED_MODULES:
                continue
            if missing_module in KNOWN_MAPS:
                missing_module = KNOWN_MAPS[missing_module]
            non_stdlib_modules += 1
            result = find_missing_optional_dependency(missing_module)
            if result is None:
                missing.append(missing_module)
        failed = len(missing) / non_stdlib_modules
        if missing:
            pytest.fail(
                f"{failed * 100:.2f}% of missing modules not found "
                f"in project dependencies: {missing}"
            )
