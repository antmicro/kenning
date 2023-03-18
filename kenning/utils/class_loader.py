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
import ast

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

    root_module = importlib.util.find_spec(modulepath)
    modules_to_parse = [root_module]
    i = 0
    # get all submodules
    while i < len(modules_to_parse):
        module = modules_to_parse[i]
        i += 1
        if '__init__' not in module.origin:
            continue
        # iterate python files
        for submodule_path in Path(module.origin).parent.glob('*.py'):
            if '__init__' == submodule_path.stem:
                continue
            modules_to_parse.append(importlib.util.find_spec(
                f'{module.name}.{submodule_path.stem}'
            ))
        # iterate subdirectories
        for submodule_path in Path(module.origin).parent.glob('*'):
            if not submodule_path.is_dir():
                continue
            module_spec = importlib.util.find_spec(
                f'{module.name}.{submodule_path.name}'
            )
            if module_spec.has_location:
                modules_to_parse.append(module_spec)

    # get all class definitions from all files
    classes_defs = dict()
    classes_modules = dict()
    for module in modules_to_parse:
        with open(module.origin, 'r') as f:
            parsed_file = ast.parse(f.read())
        for elem in parsed_file.body:
            if not isinstance(elem, ast.ClassDef):
                continue
            classes_defs[elem.name] = elem
            classes_modules[elem.name] = module

    # recursively filter subclasses
    subclasses = set()
    checked_classes = {cls.__name__}
    non_final_subclasses = set()

    def collect_subclasses(class_def: ast.ClassDef) -> bool:
        """
        Updates the set of subclasses with subclasses for a given class.

        It is an internal function updating the `subclasses`,
        `checked_classes` structures.

        Parameters
        ----------
        class_def : ast.ClassDef
            Class to collect subclasses for

        Returns
        -------
        bool : True if class_def is subclass of cls
        """
        found_subclass = False
        checked_classes.add(class_def.name)
        for b in class_def.bases:
            non_final_subclasses.add(b.id)
            if b.id == cls.__name__:
                subclasses.add(class_def.name)
                found_subclass = True
            elif b.id in subclasses or (
                    b.id in classes_defs and
                    collect_subclasses(classes_defs[b.id])):
                subclasses.add(class_def.name)
                found_subclass = True
        return found_subclass

    for class_name, class_def in classes_defs.items():
        if class_name not in checked_classes:
            collect_subclasses(class_def)

    # try importing subclasses
    result = []
    for subclass_name in subclasses:
        # filter non final subclasses
        if subclass_name in non_final_subclasses:
            continue
        subclass_module = classes_modules[subclass_name]
        try:
            subclass = getattr(
                importlib.import_module(subclass_module.name),
                subclass_name
            )
            result.append(subclass)
        except ImportError as e:
            logger.error(
                f'Could not import subclass: {subclass_name}, error: {e}'
            )
            if raise_exception:
                raise

    result.sort(key=lambda c: c.__name__)

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
