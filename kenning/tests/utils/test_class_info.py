# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import ast
import textwrap
from typing import Any

from kenning.utils.class_info import Argument, get_class_module_name


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
