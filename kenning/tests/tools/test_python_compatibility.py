# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Checks that Kenning's extras resolve on each Python version.
"""

import pytest
from check_python_compatibility import PYPROJECT, is_untested, uv_resolve
from pyproject_requirements import Pyproject

VERSIONS = ["3.11", "3.12", "3.13", "3.14", "3.15"]

# Extras known not to resolve on a Python version.
# An extra that starts resolving fails the test, so it can be removed here.
EXPECTED_FAILURES = {
    "3.14": {
        "tensorflow": "tensorflow has no cp314 wheels",
        "onnx2tf": "requires tensorflow",
        "tvm": "no tvm wheel for cp314",
        "all": "requires tensorflow and tvm",
    },
    "3.15": {
        "tensorflow": "tensorflow has no cp315 wheels",
        "onnx2tf": "requires tensorflow",
        "tvm": "no tvm wheel for cp315",
        "all": "requires tensorflow and tvm",
        "torch": "no torch wheel for cp315",
        "anomaly_detection": "requires torch",
        "auto_pytorch": "requires torch",
        "nni": "requires torch",
        "executorch": "requires torch",
        "llm": "requires torch",
        "sparsity_aware_kernel": "requires torch",
        "onnxruntime": "no onnxruntime wheel for cp315",
        "onnxruntime_gpu": "no onnxruntime-gpu wheel for cp315",
        "real_time_visualization": "no dearpygui wheel for cp315",
        "gemma": "gemma requires grain, which has no cp315 wheel",
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
