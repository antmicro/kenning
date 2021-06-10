"""
Provides methods for importing classes and modules at runtime based on string.
"""

import importlib
from typing import ClassVar, List
from pathlib import Path


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
    modulename = '.'.join(parts[parts.index('kenning'):]).rstrip('.py')
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
    command = [ar.strip() for ar in argv]
    modulename = get_kenning_submodule_from_path(command[0])
    flagpresent = False
    for i in range(len(command)):
        if command[i].startswith('-'):
            flagpresent = True
        elif flagpresent:
            command[i] = '    ' + command[i]
    command = [f'python -m {modulename} \\'] + [f'    {ar} \\' for ar in command[1:-1]] + [f'    {command[-1]}']  # noqa: E501
    return command
