# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides methods for importing classes and modules at runtime based on string.
"""

import abc
import ast
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from kenning.core.dataconverter import DataConverter
from kenning.core.dataprovider import DataProvider
from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.onnxconversion import ONNXConversion
from kenning.core.optimizer import Optimizer
from kenning.core.outputcollector import OutputCollector
from kenning.core.protocol import Protocol
from kenning.core.runner import Runner
from kenning.core.runtime import Runtime
from kenning.core.runtimebuilder import RuntimeBuilder
from kenning.utils.logger import KLogger

OPTIMIZERS = "optimizers"
RUNNERS = "runners"
DATA_PROVIDERS = "dataproviders"
DATA_CONVERTERS = "dataconverters"
DATASETS = "datasets"
MODEL_WRAPPERS = "modelwrappers"
ONNX_CONVERSIONS = "onnxconversions"
OUTPUT_COLLECTORS = "outputcollectors"
RUNTIME_BUILDERS = "runtimebuilders"
RUNTIME_PROTOCOLS = "protocols"
RUNTIMES = "runtimes"


def get_base_classes_dict() -> Dict[str, Tuple[str, Type]]:
    """
    Returns collection of Kenning groups of modules.

    Returns
    -------
    Dict[str, Tuple[str, Type]]
        dict with keys corresponding to names of
        groups of modules, values are module paths and base class names
    """
    return {
        OPTIMIZERS: ("kenning.optimizers", Optimizer),
        RUNNERS: ("kenning.runners", Runner),
        DATA_PROVIDERS: ("kenning.dataproviders", DataProvider),
        DATA_CONVERTERS: ("kenning.dataconverters", DataConverter),
        DATASETS: ("kenning.datasets", Dataset),
        MODEL_WRAPPERS: ("kenning.modelwrappers", ModelWrapper),
        ONNX_CONVERSIONS: ("kenning.onnxconverters", ONNXConversion),
        OUTPUT_COLLECTORS: ("kenning.outputcollectors", OutputCollector),
        RUNTIME_BUILDERS: ("kenning.runtimebuilders", RuntimeBuilder),
        RUNTIME_PROTOCOLS: ("kenning.protocols", Protocol),
        RUNTIMES: ("kenning.runtimes", Runtime),
    }


def get_all_subclasses(
    module_path: str,
    cls: Type,
    raise_exception: bool = False,
    import_classes: bool = True,
    show_warnings: bool = True,
) -> Union[List[Type], List[Tuple[str, str]]]:
    """
    Retrieves all subclasses of given class. Filters classes that are not
    final.

    Parameters
    ----------
    module_path : str
        Module-like path to where search should be done.
    cls : Type
        Given base class.
    raise_exception : bool
        Indicate if exception should be raised in case subclass cannot be
        imported.
    import_classes: bool
        Whether to import classes into memory or just return a list of modules
    show_warnings: bool
        Tells whether method should print warnings if modules could not be
        imported.

    Returns
    -------
    Union[List[Type], List[Tuple[str, str]]]
        When importing classes: List of all final subclasses of given class.
        When not importing classes: list of tuples with name and module path
        of the class

    Raises
    ------
    ModuleNotFoundError, ImportError
        When modules could not be imported
    Exception
        If some unspecified errors occurred during imports
    """
    root_module = importlib.util.find_spec(module_path)
    modules_to_parse = [root_module]
    i = 0
    # get all submodules
    while i < len(modules_to_parse):
        module = modules_to_parse[i]
        i += 1
        if "__init__" not in module.origin:
            continue
        # iterate python files
        for submodule_path in Path(module.origin).parent.glob("*.py"):
            if "__init__" == submodule_path.stem:
                continue
            modules_to_parse.append(
                importlib.util.find_spec(
                    f"{module.name}.{submodule_path.stem}"
                )
            )
        # iterate subdirectories
        for submodule_path in Path(module.origin).parent.glob("*"):
            if not submodule_path.is_dir():
                continue
            module_spec = importlib.util.find_spec(
                f"{module.name}.{submodule_path.name}"
            )
            if module_spec.has_location:
                modules_to_parse.append(module_spec)

    # get all class definitions from all files
    classes_defs = dict()
    classes_modules = dict()
    for module in modules_to_parse:
        with open(module.origin, "r") as f:
            parsed_file = ast.parse(f.read())
        for elem in parsed_file.body:
            if not isinstance(elem, ast.ClassDef):
                continue
            classes_defs[elem.name] = elem
            classes_modules[elem.name] = module

    # recursively filter subclasses
    subclasses = set()
    checked_classes = set()

    def collect_subclasses(class_def: ast.ClassDef) -> bool:
        """
        Updates the set of subclasses with subclasses for a given class.

        It is an internal function updating the `subclasses`,
        `checked_classes` structures.

        Parameters
        ----------
        class_def : ast.ClassDef
            Class to collect subclasses for.

        Returns
        -------
        bool
            True if class_def is subclass of cls.
        """
        found_subclass = False
        checked_classes.add(class_def.name)
        for b in class_def.bases:
            if not hasattr(b, "id"):
                continue
            if b.id == cls.__name__:
                found_subclass = True
            elif b.id in subclasses or (
                b.id in classes_defs and collect_subclasses(classes_defs[b.id])
            ):
                found_subclass = True
            if found_subclass:
                subclasses.add(class_def.name)
        return found_subclass

    for class_name, class_def in classes_defs.items():
        if class_name not in checked_classes:
            collect_subclasses(class_def)

    # try importing subclasses
    result = []
    for subclass_name in subclasses:
        subclass_module = classes_modules[subclass_name]
        try:
            if not import_classes:
                result.append((subclass_name, subclass_module.name))
                continue

            subclass = getattr(
                importlib.import_module(subclass_module.name), subclass_name
            )
            # filter abstract classes
            if (
                not inspect.isabstract(subclass)
                and abc.ABC not in subclass.__bases__
            ):
                result.append(subclass)
        except (ModuleNotFoundError, ImportError, Exception) as e:
            if show_warnings:
                msg = f"Could not add {subclass_name}. Reason:"
                KLogger.warning("-" * len(msg))
                KLogger.warning(msg)
                KLogger.warning(e)
                KLogger.warning("-" * len(msg))
            if raise_exception:
                raise

    if import_classes:
        result.sort(key=lambda c: c.__name__)

    return result


def any_from_json(json_cfg: Dict[str, Any], **kwargs) -> Optional[Any]:
    """
    Loads the class using `from_json` method, if available.

    Parameters
    ----------
    json_cfg: Dict[str, Any]
        A JSON object snippet with `type` parameter, specifying the
        full name of the class, and `parameters` parameter, with list
        of constructor arguments for the class.
    **kwargs:
        Additional arguments

    Returns
    -------
    Optional[Any]
        If a class is available and contains `from_json` method, it
        returns object of this class.
    """
    if "type" not in json_cfg or "parameters" not in json_cfg:
        return None
    cls = load_class(json_cfg["type"])
    if not hasattr(cls, "from_json"):
        return None
    return cls.from_json(json_cfg["parameters"], **kwargs)


def load_class(module_path: str) -> Type:
    """
    Loads class given in the module path.

    Parameters
    ----------
    module_path : str
        Module-like path to the class.

    Returns
    -------
    Type
        Loaded class.
    """
    module_name, cls_name = module_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls


def get_kenning_submodule_from_path(module_path: str) -> str:
    """
    Converts script path to kenning submodule name.

    Parameters
    ----------
    module_path : str
        Path to the module script, usually stored in sys.argv[0].

    Returns
    -------
    str
        Normalized module path.
    """
    parts = Path(module_path).parts
    item_index = len(parts) - 1 - parts[::-1].index("kenning")
    modulename = ".".join(parts[item_index:]).rstrip(".py")
    return modulename


def get_command(argv: List[str] = None, with_slash: bool = True) -> List[str]:
    """
    Creates a string with command.

    Parameters
    ----------
    argv : List[str]
        List or arguments from sys.argv.
    with_slash : bool
        Tells if slash should be included in command

    Returns
    -------
    List[str]
        Full string with command.
    """
    if argv is None:
        argv = sys.argv
    command = [ar.strip() for ar in argv if ar.strip() != ""]

    modulename = None
    if not str(Path(command[0]).resolve()).endswith("kenning"):
        modulename = get_kenning_submodule_from_path(command[0])

    flagpresent = False
    first_flag = 1
    for i in range(len(command)):
        if command[i].startswith("-"):
            if not flagpresent:
                first_flag = i
            flagpresent = True
        elif flagpresent:
            command[i] = "    " + command[i]

    if modulename:
        result = [f"python -m {modulename}"]
        first_flag = 1
    else:
        result = [f"kenning {' '.join(command[1:first_flag])}"]

    if len(command) > 1:
        result[0] = f"{result[0]} " + ("\\" if with_slash else "")
        result += [
            f"    {ar} " + ("\\" if with_slash else "")
            for ar in command[first_flag:-1]
        ] + [f"    {command[-1]}"]
    return result
