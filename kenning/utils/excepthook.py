# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with custom import function and script matching missing modules
with pyproject dependencies.
"""

import re
import sys
from types import TracebackType
from typing import Optional, Type

from kenning.resources.pyproject.dependencies import get_all_dependencies

# Regexes for comparing dependencies
END_CHARS = " >=<@~"
RE_NAME = f"^[^{END_CHARS}]+(?=[{END_CHARS}]?)"
RE_KENNING_DEP = r"kenning\[[a-zA-Z]+\]"

# Mapping of names used for import and oficiall package names
KNOWN_PYPI_NAMES = {
    "cv2": "opencv_python",
    "tvm": "apache-tvm",
}


class MissingKenningDependencies(ModuleNotFoundError):
    """
    Extension of ModuleNotFoundError with information
    about optional dependencies.
    """

    # Group of optional dependencies (from pyproject.toml) or 'kenning'
    optional_dependencies: str

    def __init__(
        self,
        *args: object,
        name: Optional[str] = None,
        path: Optional[str] = None,
        optional_dependencies: Optional[str] = None,
    ):
        super().__init__(*args, name=name, path=path)
        self.optional_dependencies = optional_dependencies

    def __str__(self) -> str:
        if self.optional_dependencies == "kenning":
            return "Required module is missing, please reinstall Kenning.\n"
        else:
            return (
                "This method requires additional dependencies, please use"
                f' `pip install "kenning[{self.optional_dependencies}]"` to install them.\n'  # noqa: E501
            )


def kenning_missing_import_excepthook(
    type_: Type[BaseException],
    value: BaseException,
    traceback: TracebackType,
):
    """
    Extended `sys.excepthook`, which for ModuleNotFoundError tries to find
    Kenning optional dependencies containing missing module.

    Parameters
    ----------
    type_ : Type[BaseException]
        Type of the raised exception
    value : BaseException
        The raised exception
    traceback : TracebackType
        Traceback of the raised exception
    """
    # Get last step of traceback
    last_tb = traceback
    while last_tb.tb_next:
        last_tb = last_tb.tb_next
    # If exception is raised in Kenning try to find
    # which optional dependency should be installed
    if type_ == ModuleNotFoundError and last_tb.tb_frame.f_globals.get(
        "__package__", ""
    ).startswith("kenning."):
        extras = find_missing_optional_dependency(value.name)
        if extras:
            type_ = MissingKenningDependencies
            value = type_(
                name=value.name, path=value.path, optional_dependencies=extras
            )
    sys.__excepthook__(type_, value, traceback)


def _get_pypi_name(name: str) -> str:
    """
    Tries to convert module's name used in code to proper name used by `pip`.

    Parameters
    ----------
    name : str
        Name of module used by `import`

    Returns
    -------
    str :
        Name of module used by `pip`
    """
    return KNOWN_PYPI_NAMES.get(name, name)


def _normalize(name: str) -> str:
    """
    Normalizes string.

    Replaces '-' with '_' and ensures only lower-case letters are used.

    Parameter
    ---------
    name : str
        Any string

    Return
    ------
    str :
        Normalized string
    """
    return name.replace("-", "_").lower()


def find_missing_optional_dependency(module_name: str) -> Optional[str]:
    """
    Checks which group of optional dependencies contains missing module.

    Parameters
    ----------
    module_name : str
        Name of the missing module

    Returns
    -------
    Optional[str] :
        Name of group with optional dependencies which contain missing module
        or None otherwise.
    """
    found = None
    dependencies = get_all_dependencies()
    for extra, modules in dependencies.items():
        re.match(RE_NAME, extra).group(0)
        if any(
            _normalize(_get_pypi_name(module_name))
            in _normalize(re.match(RE_NAME, name).group(0))
            for name in modules
            if not re.match(RE_KENNING_DEP, name)
        ):
            found = extra
            break

    return found
