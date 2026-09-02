# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Checks the weight of packages imported with kenning.cli.cli, which is loaded
on every TAB press when autocomplete is enabled. It aims to prevent the case
when it gets too slow. If this test fails, consider lazy loading heavy
packages.
"""


import json
import subprocess
import sys
from typing import Dict

ENTRYPOINT = "kenning.cli.cli"
MAX_PACKAGE_SIZE_MB = 15

PROBE = """
import json, os, sys

import {module}

sizes = {{}}
for name, imported in list(sys.modules.items()):
    path = getattr(imported, "__file__", None)
    if not path:
        continue
    try:
        size = os.path.getsize(path)
    except OSError:
        continue
    package = name.split(".")[0]
    sizes[package] = sizes.get(package, 0) + size

print(json.dumps(sizes))
"""


def _measure_imported_sizes(module: str) -> Dict[str, float]:
    """
    Measures how much code importing a module pulls in, per top-level package.

    Parameters
    ----------
    module : str
        Name of the module to import.

    Returns
    -------
    Dict[str, float]
        Size in megabytes of each imported top-level package.
    """
    sizes = subprocess.run(
        [sys.executable, "-c", PROBE.format(module=module)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    return {package: size / 1e6 for package, size in json.loads(sizes).items()}


def test_no_heavy_package_imported_by_cli():
    sizes = _measure_imported_sizes(ENTRYPOINT)

    too_heavy = {
        package: size
        for package, size in sizes.items()
        if size > MAX_PACKAGE_SIZE_MB
    }

    def _summary(measured):
        return "\n".join(
            f"  {size:7.1f} MB  {package}"
            for package, size in sorted(
                measured.items(), key=lambda item: -item[1]
            )
        )

    assert not too_heavy, (
        f"Packages imported by {ENTRYPOINT} exceed the "
        f"{MAX_PACKAGE_SIZE_MB} MB budget:\n"
        f"{_summary(too_heavy)}"
    )
