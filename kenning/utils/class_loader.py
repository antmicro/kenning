# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides methods for importing classes and modules at runtime based on string.
"""

from typing import Type
import importlib
from typing import ClassVar, List
from pathlib import Path
import pkgutil

from kenning.utils.logger import get_logger


def get_all_subclasses(
        modulepath: str,
        cls: Type,
        raise_exception: bool = False) -> List[Type]:
    """
    Retrieves all subclasses of given class. Filters classes that are not
    final.

    Parameters
    ----------
    modulepath : str
        Module-like path to where search should be done
    cls : Type
        Given base class
    raise_exception : bool
        Indicate if exception should be raised in case subclass cannot be
        imported

    Returns
    -------
    List[Type] :
        List of all final subclasses of given class
    """
    logger = get_logger()

    result = []
    queue = [pkgutil.resolve_name(modulepath)]
    while queue:
        q = queue.pop()
        prefix = q.__name__ + '.'
        for m in pkgutil.iter_modules(q.__path__, prefix):
            try:
                module = pkgutil.resolve_name(m.name)
                if m.ispkg:
                    queue.append(module)
                else:
                    result.append(module)
            except Exception:
                if raise_exception:
                    logger.error(f'Could not import module: {m}')
                    raise
                else:
                    logger.warn(f'Could not import module: {m}')

    result = []
    queue = [cls]
    while queue:
        q = queue.pop()
        if len(q.__subclasses__()) == 0:
            result.append(q)
        for sub_q in q.__subclasses__():
            queue.append(sub_q)
    return result


def load_class(modulepath: str) -> ClassVar:
    """
    Loads class given in the module path.

    Parameters
    ----------
    modulepath : str
        Module-like path to the class
    """
    module_name, cls_name = modulepath.rsplit('.', 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls


def get_kenning_submodule_from_path(modulepath: str):
    """
    Converts script path to kenning submodule name.

    Parameters
    ----------
    modulepath: str
        Path to the module script, usually stored in sys.argv[0]

    Returns
    -------
    str: Normalized module path
    """
    parts = Path(modulepath).parts
    item_index = len(parts) - 1 - parts[::-1].index("kenning")
    modulename = '.'.join(parts[item_index:]).rstrip('.py')
    return modulename


def get_command(argv: List[str]):
    """
    Creates a string with command.

    Parameters
    ----------
    argv: List[str]
        List or arguments from sys.argv

    Returns
    -------
    str: Full string with command
    """
    command = [ar.strip() for ar in argv if ar.strip() != '']
    modulename = get_kenning_submodule_from_path(command[0])
    flagpresent = False
    for i in range(len(command)):
        if command[i].startswith('-'):
            flagpresent = True
        elif flagpresent:
            command[i] = '    ' + command[i]
    if (len(command) > 1):
        command = [f'python -m {modulename} \\'] + [f'    {ar} \\' for ar in command[1:-1]] + [f'    {command[-1]}']  # noqa: E501
    else:
        command = [f'python -m {modulename}']
    return command
