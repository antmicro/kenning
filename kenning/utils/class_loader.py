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

    result = []
    queue = [pkgutil.resolve_name(modulepath)]
    module_classes = {}
    module_import_error = {}
    # BFS over modules
    while queue:
        module = queue.pop()
        prefix = module.__name__ + '.'
        # iterate submodules
        for sub_module in pkgutil.iter_modules(module.__path__, prefix):
            # get list of classes in module
            file_path = pkgutil.get_loader(sub_module.name).path
            with open(file_path, 'r') as mod_file:
                parsed_file = ast.parse(mod_file.read())
            module_classes[sub_module.name] = []
            for c in parsed_file.body:
                if not isinstance(c, ast.ClassDef):
                    continue
                module_classes[sub_module.name].append(c)
            # try import
            try:
                sub_module_name = pkgutil.resolve_name(sub_module.name)
                if sub_module.ispkg:
                    queue.append(sub_module_name)
                else:
                    result.append(sub_module_name)
            except Exception as e:
                module_import_error[sub_module.name] = e

    # log warn message with unimported classes
    for module_name in module_classes.keys():
        if module_name not in module_import_error.keys():
            continue
        logger.warn(
            f'Could not import module {module_name}, skipped classes: '
            f'{[c.name for c in module_classes[module_name]]}'
        )

    # BFS over classes
    result = []
    all_classes = []
    queue = [cls]
    while queue:
        q = queue.pop()
        all_classes.append(q)
        if len(q.__subclasses__()) == 0:
            result.append(q)
        for sub_q in q.__subclasses__():
            queue.append(sub_q)

    # get subclasses that could not be imported
    if raise_exception:
        not_imported_subclasses = []
        all_classes_names = [r.__name__ for r in all_classes]
        # iterate over modules with errors
        for module, _ in module_import_error.items():
            for class_def in module_classes[module]:
                # if any base class is in all_classes
                class_def_bases = [b.id for b in class_def.bases]
                if len(set(all_classes_names) & set(class_def_bases)):
                    not_imported_subclasses.append(class_def.name)
        if len(not_imported_subclasses):
            logger.error(
                f'Could not import subclasses: {not_imported_subclasses}'
            )
            raise ImportError

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
