# Copyright (c) 2020-2024 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import ast
import sys
from pathlib import Path
from typing import Generator, List, Tuple

import pytest

import kenning
from kenning.utils.class_loader import (
    get_kenning_submodule_from_path,
    load_class,
)

KENNING_PYTHON_FILES = list(Path(kenning.__file__).parent.rglob("*.py"))


def get_all_classes(
    python_files: List[str]
) -> Generator[Tuple[ast.ClassDef, str], None, None]:
    """
    Generator returning every Class defined in files.

    Parameters
    ----------
    python_files : List[str]
        List of files with defined classes

    Yields
    ------
    ast.ClassDef
        Definition of the class
    str
        Path of the file containing found class
    """
    for file_path in python_files:
        if "/kenning/tests/" in str(file_path):
            continue
        with open(file_path, "r") as fd:
            content = fd.read()
        file_ast = ast.parse(
            content,
            feature_version=sys.version_info[:2],
        )
        for node in ast.walk(file_ast):
            if isinstance(node, ast.ClassDef):
                yield node, file_path


def get_all_methods(
    class_ast: ast.ClassDef,
) -> Generator[ast.FunctionDef, None, None]:
    """
    Generator returning every function defined in class.

    Parameters
    ----------
    class_ast : ast.ClassDef
        Definition of the class

    Yields
    ------
    ast.FunctionDef
        Definition of function
    """
    for node in ast.walk(class_ast):
        if isinstance(node, ast.FunctionDef):
            yield node


def check_class_bases(class_ast: ast.ClassDef) -> bool:
    """
    Checks if classes is based on ABC - abstract class.

    Parameters
    ----------
    class_ast : ast.ClassDef
        Definistion of the class

    Returns
    -------
    bool
        Is call is abstract or not
    """
    return any(
        (base.attr if isinstance(base, ast.Attribute) else base.id) == "ABC"
        for base in class_ast.bases
    )


class TestAbstract:
    @pytest.mark.parametrize(
        "class_ast,file_path",
        [
            pytest.param(
                class_ast,
                file_path,
            )
            for class_ast, file_path in get_all_classes(KENNING_PYTHON_FILES)
        ],
    )
    def test_abstract_methods(
        self,
        class_ast: ast.ClassDef,
        file_path: str,
    ):
        """
        Test checking if @abstractmethod is defined in ABC class.
        """

        def isabstractmethod(decorator):
            if hasattr(decorator, "id"):
                return decorator.id == "abstractmethod"
            elif hasattr(decorator, "func") and hasattr(decorator.func, "id"):
                return decorator.func.id == "abstractmethod"
            return False

        if not check_class_bases(class_ast) and any(
            any(
                isabstractmethod(decorator)
                for decorator in method_ast.decorator_list
            )
            for method_ast in get_all_methods(class_ast)
        ):
            pytest.fail(
                msg=f"Class {class_ast.name}: {file_path} is not marked "
                "as abstract, despite having abstract methods"
            )

    @pytest.mark.parametrize(
        "class_ast,file_path",
        [
            pytest.param(
                class_ast,
                file_path,
            )
            for class_ast, file_path in get_all_classes(KENNING_PYTHON_FILES)
        ],
    )
    def test_class_contains_abstract_methods(self, class_ast, file_path):
        """
        Test checking if non-abstract class implements all abstract methods
        from its base classes.
        """
        if check_class_bases(class_ast):
            return
        rel_path = Path(file_path).relative_to(Path(kenning.__file__).parent)
        module = get_kenning_submodule_from_path(file_path)
        if "kenning.onnxconverters" in module:
            pytest.xfail("Module kenning.onnxconverters is a legacy code")
        try:
            cls = load_class(f"{module}.{class_ast.name}")
        except ModuleNotFoundError as e:
            pytest.skip(
                f"Class {class_ast.name} ({rel_path}) cannot be imported "
                f"due to the missing module: {e}"
            )
        except AttributeError:
            pytest.skip(
                f"Class {class_ast.name} ({rel_path}) is hidden inside"
                " other class or function"
            )
        if hasattr(cls, "__abstractmethods__") and cls.__abstractmethods__:
            pytest.fail(
                msg=f"Class {class_ast.name} ({rel_path}) is not marked "
                "as abstract and does not implement all abstract method:"
                f" {list(cls.__abstractmethods__)}"
            )
