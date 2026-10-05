# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import ast
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Tuple

import pytest

from kenning.utils import class_info
from kenning.utils.class_info import (
    Argument,
    evaluate_argument_list,
    evaluate_argument_list_of_keys,
    get_args_structure_from_parameterschema,
    get_class_module_docstrings,
    get_class_module_name,
    get_dependency,
    get_input_specification,
    get_io_specification,
    get_output_specification,
    parse_io_dict_node_to_string,
    parse_io_spec_dict_to_str,
)

IO_SPECIFICATION_CLASS = """
class SampleModelWrapper:
    @classmethod
    def _get_io_specification(cls, num_classes, batch_size=1):
        return {
            "input": [
                {
                    "name": "input_1",
                    "shape": (batch_size, 3, 224, 224),
                    "dtype": "float32",
                }
            ],
            "output": [
                {
                    "name": OUTPUT_NAME,
                    "shape": (batch_size, num_classes),
                    "dtype": "float32",
                }
            ],
            "processed_output": [],
        }
"""

IO_SPECIFICATION_LINES = [
    "### input\n\n",
    "* `input_1`\n",
    "    * `shape`: `(batch_size, 3, 224, 224)`\n",
    "    * `dtype`: `float32`\n",
    "### output\n\n",
    "* `OUTPUT_NAME`\n",
    "    * `shape`: `(batch_size, num_classes)`\n",
    "    * `dtype`: `float32`\n",
]

EVALUATE_ARGUMENT_LIST_MODULE = """
OTHER = ["unused"]
obj.attribute = ["unused"]
first, second = ["unused"], ["unused"]
MODES = ["fast", "slow"]
PRECISIONS = [8, 16]
RESOURCES = {"small": 1, "large": 2}
"""


@pytest.fixture
def evaluate_argument_list_source(tmp_path: Path) -> str:
    """
    Writes a module with sample list and dict assignments to a file and
    returns the path to it.
    """
    source_path = tmp_path / "sample_module.py"
    source_path.write_text(EVALUATE_ARGUMENT_LIST_MODULE)

    return str(source_path)


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
        """
        Argument with only a name renders as a single bullet.
        """
        argument = Argument()
        argument.name = "model_path"

        assert repr(argument) == "* `model_path`"

    def test_repr_all_fields(self):
        """
        Argument with all fields set renders every field in order.
        """
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
        """
        Class node renders as a heading without its docstring.
        """
        node = parse_node(
            """
            class SampleBlock:
                \"\"\"Docstring is not included.\"\"\"
            """
        )

        assert get_class_module_name(node) == "# SampleBlock\n\n"


class TestGetClassModuleDocstrings:
    def test_class_without_docstring(self, fake_pandoc):
        """
        Class without a docstring renders only a heading, skipping pandoc.
        """
        node = parse_node(
            """
            class SampleBlock:
                pass
            """
        )

        assert get_class_module_docstrings(node) == "# Class SampleBlock\n\n"
        assert fake_pandoc == []

    def test_class(self, fake_pandoc):
        """
        Class docstring is converted from RST to Markdown under a heading.
        """
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
        """
        Module docstring is converted under a module description heading.
        """
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
        """
        Standard library and Kenning imports are not listed.
        """
        assert get_dependency(parse_node(source)) == ""

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("import numpy", "* `numpy`\n"),
            ("from numpy import array", "* `numpy.array`\n"),
        ],
    )
    def test_available(self, source: str, expected: str):
        """
        Installed third-party imports are listed by their full path.
        """
        assert get_dependency(parse_node(source)) == expected

    def test_missing(self):
        """
        Missing imports are listed together with the import error.
        """
        result = get_dependency(
            parse_node("import class_info_missing_dependency")
        )

        assert result == (
            "* `class_info_missing_dependency` - Not available (Reason: "
            "No module named 'class_info_missing_dependency')\n"
        )


class TestGetInputSpecification:
    def test_empty_list(self):
        """
        Empty input types produce no output.
        """
        assert get_input_specification(parse_node("inputtypes = []")) == ""

    def test_list(self):
        """
        Each input type from a list is rendered as a bullet.
        """
        node = parse_node('inputtypes = ["keras", "onnx"]')

        assert get_input_specification(node) == "* `keras`\n* `onnx`\n"

    def test_dict(self):
        """
        Each key of an input types dict is rendered as a bullet.
        """
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
        """
        Empty output types return None.
        """
        assert get_output_specification(parse_node("outputtypes = []")) is None

    def test_list(self):
        """
        Each output type is rendered as a bullet.
        """
        node = parse_node('outputtypes = ["tflite"]')

        assert get_output_specification(node) == "* `tflite`\n"


class TestParseIODictNodeToString:
    def test_not_a_dict(self):
        """
        Non-dict node results in an empty line.
        """
        node = parse_node('["input", "output"]').value

        assert parse_io_dict_node_to_string(node) == [""]

    def test_io_dict(self):
        """
        IO specification dict node is rendered per tensor, skipping empty keys.
        """
        class_node = parse_node(IO_SPECIFICATION_CLASS)
        dict_node = class_node.body[0].body[0].value

        assert (
            parse_io_dict_node_to_string(dict_node) == IO_SPECIFICATION_LINES
        )


class TestGetIOSpecification:
    def test_io_specification(self):
        """
        Dict returned directly from _get_io_specification is rendered.
        """
        class_node = parse_node(IO_SPECIFICATION_CLASS)

        assert get_io_specification(class_node) == IO_SPECIFICATION_LINES

    @pytest.mark.parametrize(
        "source",
        [
            """
            class SampleBlock:
                inputtypes = ["onnx"]
            """,
            """
            class SampleBlock:
                def get_io_specification(self):
                    return {"input": []}
            """,
            """
            class SampleBlock:
                def _get_io_specification(self):
                    raise NotImplementedError
            """,
        ],
        ids=["no-methods", "other-method", "no-return"],
    )
    def test_no_io_specification(self, source: str):
        """
        Classes without a returned _get_io_specification dict produce nothing.
        """
        assert get_io_specification(parse_node(source)) == []

    def test_returned_variable(self):
        """
        Dict assigned to a variable and then returned is resolved and rendered.
        """
        class_node = parse_node(
            """
            class SampleModelWrapper:
                @classmethod
                def _get_io_specification(cls, num_classes, batch_size=1):
                    io_spec = {
                        "input": [
                            {
                                "name": "input_1",
                                "shape": (batch_size, 3, 224, 224),
                                "dtype": "float32",
                            }
                        ],
                        "output": [
                            {
                                "name": OUTPUT_NAME,
                                "shape": (batch_size, num_classes),
                                "dtype": "float32",
                            }
                        ],
                        "processed_output": [],
                    }
                    if num_classes > 1:
                        io_spec["output"][0]["class_names"] = ["a", "b"]
                    return io_spec
            """
        )

        assert get_io_specification(class_node) == IO_SPECIFICATION_LINES

    def test_returned_parameter(self):
        """
        Returned variable that is not a dict assignment produces nothing.
        """
        class_node = parse_node(
            """
            class SampleRunner:
                @classmethod
                def _get_io_specification(cls, model_io_spec):
                    return model_io_spec
            """
        )

        assert get_io_specification(class_node) == []


class TestEvaluateArgumentListOfKeys:
    def test_keys(self, evaluate_argument_list_source):
        """
        Keys of a dict variable are returned with their list type.
        """
        assert evaluate_argument_list_of_keys(
            "RESOURCES", evaluate_argument_list_source
        ) == (["small", "large"], "List[str]")

    def test_missing(self, evaluate_argument_list_source):
        """
        Undefined variable results in no keys and no type.
        """
        assert evaluate_argument_list_of_keys(
            "MISSING", evaluate_argument_list_source
        ) == ([], "")


class TestEvaluateArgumentList:
    @pytest.mark.parametrize(
        "argument_list,expected",
        [
            ("MODES", (["fast", "slow"], "List[str]")),
            ("PRECISIONS", ([8, 16], "List[int]")),
            ('["fast", "slow"]', (["fast", "slow"], "List[str]")),
            ("[]", ([], "List[]")),
            ("MISSING", ([], "")),
        ],
        ids=["variable", "int-variable", "inline-list", "empty", "missing"],
    )
    def test_evaluate(
        self,
        evaluate_argument_list_source,
        argument_list: str,
        expected: Tuple[List, str],
    ):
        """
        Lists from variables or inline literals are evaluated with their type.
        """
        assert (
            evaluate_argument_list(
                argument_list, evaluate_argument_list_source
            )
            == expected
        )

    def test_unresolvable(self, evaluate_argument_list_source):
        """
        Inline list referencing undefined names returns an error.
        """
        enum_elements, argument_type = evaluate_argument_list(
            "[Mode.FAST]", evaluate_argument_list_source
        )

        assert argument_type == "Error"
        assert enum_elements[0].startswith("Static code analysis failed")


class TestGetArgsStructureFromParameterschema:
    def test_empty(self):
        """
        Schema without properties results in an empty line.
        """
        assert get_args_structure_from_parameterschema({"properties": {}}) == [
            ""
        ]

    def test_minimal(self):
        """
        Argument without fields renders its name and argparse flag.
        """
        assert get_args_structure_from_parameterschema(
            {"properties": {"model_path": {}}}
        ) == [
            "### `model_path`\n",
            "* argparse flag: `--model-path`\n",
        ]

    def test_all_fields(self):
        """
        All schema fields are rendered, with lists and classes expanded.
        """
        parameterschema = {
            "properties": {
                "model_path": {
                    "real_name": "model_path",
                    "description": "Path to the model",
                    "type": ["string"],
                    "enum": ["a.onnx", "b.onnx"],
                    "class": Path,
                    "default": None,
                },
            },
            "required": ["model_path"],
        }

        assert get_args_structure_from_parameterschema(parameterschema) == [
            "### `model_path`\n",
            "* Path to the model\n",
            "* argparse flag: `--model-path`\n",
            "* type: `string`\n",
            "* enum\n",
            "    * `a.onnx`\n",
            "    * `b.onnx`\n",
            f"* class: `{Path.__module__}.{Path.__qualname__}`\n",
            "* default: `None`\n",
            "* required: `True`\n",
        ]


class TestParseIOSpecDictToStr:
    def test_not_a_dict(self):
        """
        Non-dict input results in an empty line.
        """
        assert parse_io_spec_dict_to_str(["input"]) == [""]

    def test_empty(self):
        """
        Empty IO specification produces no lines.
        """
        assert parse_io_spec_dict_to_str({}) == []

    def test_io_spec(self):
        """
        Headings are rendered first, followed by each tensor and its fields.
        """
        io_spec = {
            "input": [
                {"name": "input_1", "shape": [1, 3], "dtype": "float32"}
            ],
            "output": [
                {"name": "output_1", "dtype": "float32"},
                {"name": "output_2", "dtype": "int8"},
            ],
            "processed_output": [],
        }

        assert parse_io_spec_dict_to_str(io_spec) == [
            "### input\n\n",
            "### output\n\n",
            "### processed_output\n\n",
            "* `input_1`\n",
            "    * `shape`\n",
            "        * `1`\n",
            "        * `3`\n",
            "    * `dtype`: `float32`\n",
            "* `output_1`\n",
            "    * `dtype`: `float32`\n",
            "* `output_2`\n",
            "    * `dtype`: `int8`\n",
        ]

    def test_non_list_value(self):
        """
        Non-list values are rendered inline without being iterated.
        """
        assert parse_io_spec_dict_to_str({"input": "float32"}) == [
            "### input\n\n",
            "* `input`: `float32`\n",
        ]
