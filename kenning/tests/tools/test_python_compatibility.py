# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Checks that Kenning's extras resolve on each Python version.
"""

import pytest
from check_python_compatibility import PYPROJECT, is_untested, uv_resolve
from pyproject_requirements import Pyproject

VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]

# Extras known not to resolve on a Python version.
# An extra that starts resolving fails the test, so it can be removed here.
EXPECTED_FAILURES = {
    "3.10": {
        "tinygrad": "tinygrad requires Python 3.11 or newer",
    },
    "3.14": {
        "tensorflow": "tensorflow has no cp314 wheels",
        "onnx2tf": "requires tensorflow",
        "tvm": "no tvm wheel for cp314",
        "all": "requires tensorflow and tvm",
    },
}

PROJECT = Pyproject.load(PYPROJECT)
EXTRAS = PROJECT.get_extras()


def cases():
    for version in VERSIONS:
        expected = EXPECTED_FAILURES.get(version, {})
        for extra in EXTRAS:
            marks = []
            if extra in expected:
                marks.append(
                    pytest.mark.xfail(reason=expected[extra], strict=True)
                )
            yield pytest.param(
                version, extra, marks=marks, id=f"py{version}-{extra}"
            )


@pytest.mark.python_compat
@pytest.mark.parametrize(("version", "extra"), list(cases()))
def test_extra_resolves(version: str, extra: str):
    requirements = EXTRAS[extra]
    assert not is_untested(
        requirements, EXTRAS[PROJECT.name], version
    ), f"No package of {extra} has a requirement for Python {version}"

    error = uv_resolve(
        PYPROJECT, version, extra=None if extra == PROJECT.name else extra
    )
    assert error is None, error
