# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import ast
import textwrap
from types import SimpleNamespace
from typing import Any, List, Tuple

import pytest

from kenning.utils import class_info
from kenning.utils.class_info import (
    Argument,
    get_class_module_docstrings,
    get_class_module_name,
    get_dependency,
    get_input_specification,
    get_output_specification,
)


@pytest.fixture
def fake_pandoc(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[str, str]]:
    """
    Replaces pandoc with a fake that returns the text unchanged.
    """
    calls = []

    def read(text: str, format: str) -> str:
        calls.append(("read", format))
        return text

    def write(doc: str, format: str) -> str:
        calls.append(("write", format))
        return doc

    monkeypatch.setattr(
        class_info, "pandoc", SimpleNamespace(read=read, write=write)
    )
    return calls


def parse_node(source: str) -> Any:
    """
    Parses the source and returns its first top-level statement.
    """
    return ast.parse(textwrap.dedent(source)).body[0]


class TestArgument:
    def test_repr_only_name(self):
        argument = Argument()
        argument.name = "model_path"

        assert repr(argument) == "* `model_path`"

    def test_repr_all_fields(self):
        argument = Argument()
        argument.name = "precision"
        argument.description = "Model precision"
        argument.argparse_name = "--precision"
        argument.type = "str"
        argument.required = "True"
        argument.default = "int8"
        argument.nullable = "False"
        argument.enum = ["int8", "float32"]

        lines = repr(argument).split("\n")

        assert lines == [
            "* `precision`",
            "* description: Model precision",
            "* argparse flag: `--precision`",
            "* type: `str`",
            "* required: `True`",
            "* default value: `int8`",
            "* Can be undefined: `False`",
            "* `allowed values`:",
            "    * `int8`",
            "    * `float32`",
        ]


class TestGetClassModuleName:
    def test_class(self):
        node = parse_node(
            """
            class SampleBlock:
                \"\"\"Docstring is not included.\"\"\"
            """
        )

        assert get_class_module_name(node) == "# SampleBlock\n\n"


class TestGetClassModuleDocstrings:
    def test_class_without_docstring(self, fake_pandoc):
        node = parse_node(
            """
            class SampleBlock:
                pass
            """
        )

        assert get_class_module_docstrings(node) == "# Class SampleBlock\n\n"
        assert fake_pandoc == []

    def test_class(self, fake_pandoc):
        node = parse_node(
            """
            class SampleBlock:
                \"\"\"
                    Sample block docstring.
                \"\"\"
            """
        )

        assert (
            get_class_module_docstrings(node)
            == "# Class SampleBlock\n\nSample block docstring.\n\n"
        )
        assert fake_pandoc == [("read", "rst"), ("write", "gfm")]

    def test_module(self, fake_pandoc):
        node = ast.parse('"""\nSample module docstring.\n"""\n')

        assert (
            get_class_module_docstrings(node)
            == "# Module description\nSample module docstring.\n"
        )
        assert fake_pandoc == [("read", "rst"), ("write", "gfm")]


class TestGetDependency:
    @pytest.mark.parametrize(
        "source",
        [
            "import os",
            "from pathlib import Path",
            "import kenning",
            "from kenning.core.dataset import Dataset",
        ],
    )
    def test_skipped(self, source: str):
        assert get_dependency(parse_node(source)) == ""

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("import numpy", "* `numpy`\n"),
            ("from numpy import array", "* `numpy.array`\n"),
        ],
    )
    def test_available(self, source: str, expected: str):
        assert get_dependency(parse_node(source)) == expected

    def test_missing(self):
        result = get_dependency(
            parse_node("import class_info_missing_dependency")
        )

        assert result == (
            "* `class_info_missing_dependency` - Not available (Reason: "
            "No module named 'class_info_missing_dependency')\n"
        )


class TestGetInputSpecification:
    def test_empty_list(self):
        assert get_input_specification(parse_node("inputtypes = []")) == ""

    def test_list(self):
        node = parse_node('inputtypes = ["keras", "onnx"]')

        assert get_input_specification(node) == "* `keras`\n* `onnx`\n"

    def test_dict(self):
        node = parse_node(
            """
            inputtypes = {
                "keras": load_keras_model,
                "onnx": load_onnx_model,
            }
            """
        )

        assert get_input_specification(node) == "* `keras`\n* `onnx`\n"


class TestGetOutputSpecification:
    def test_empty_list(self):
        assert get_output_specification(parse_node("outputtypes = []")) is None

    def test_list(self):
        node = parse_node('outputtypes = ["tflite"]')

        assert get_output_specification(node) == "* `tflite`\n"
