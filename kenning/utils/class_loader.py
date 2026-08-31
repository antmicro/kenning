# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides methods for importing classes and modules at runtime based on string.
"""

import abc
import argparse
import ast
import importlib
import importlib.util
import inspect
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from kenning.cli.parser import ParserHelpException
from kenning.core.automl import AutoML
from kenning.core.converter import ModelConverter
from kenning.core.dataconverter import DataConverter
from kenning.core.dataprovider import DataProvider
from kenning.core.dataset import Dataset
from kenning.core.inferenceloop import InferenceLoop
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.core.outputcollector import OutputCollector
from kenning.core.platform import Platform
from kenning.core.protocol import Protocol
from kenning.core.report import Report
from kenning.core.runner import Runner
from kenning.core.runtime import Runtime
from kenning.core.runtimebuilder import RuntimeBuilder
from kenning.dispatcher.block_config import (
    AUTOML,
    BLOCK_CONFIGURATIONS_KEY,
    CONVERTERS,
    DATA_CONVERTERS,
    DATA_PROVIDERS,
    DATASETS,
    INFERENCE_LOOPS,
    MODEL_WRAPPERS,
    OPTIMIZERS,
    OUTPUT_COLLECTORS,
    PLATFORMS,
    REPORT,
    RUNNERS,
    RUNTIME_BUILDERS,
    RUNTIME_PROTOCOLS,
    RUNTIMES,
    ConfigKey,
    KenningBlockConfigDict,
)
from kenning.utils.logger import KLogger


def get_base_classes_dict() -> Dict[str, Tuple[str, Type]]:
    """
    Returns collection of Kenning groups of modules.

    Returns
    -------
    Dict[str, Tuple[str, Type]]
        Dict with keys corresponding to names of groups of modules, values are
        module paths and base classes.
    """
    return {
        OPTIMIZERS: ("kenning.optimizers", Optimizer),
        RUNNERS: ("kenning.runners", Runner),
        DATA_PROVIDERS: ("kenning.dataproviders", DataProvider),
        DATA_CONVERTERS: ("kenning.dataconverters", DataConverter),
        DATASETS: ("kenning.datasets", Dataset),
        MODEL_WRAPPERS: ("kenning.modelwrappers", ModelWrapper),
        OUTPUT_COLLECTORS: ("kenning.outputcollectors", OutputCollector),
        PLATFORMS: ("kenning.platforms", Platform),
        RUNTIME_BUILDERS: ("kenning.runtimebuilders", RuntimeBuilder),
        INFERENCE_LOOPS: ("kenning.inferenceloops", InferenceLoop),
        RUNTIME_PROTOCOLS: ("kenning.protocols", Protocol),
        RUNTIMES: ("kenning.runtimes", Runtime),
        AUTOML: ("kenning.automl", AutoML),
        REPORT: ("kenning.report", Report),
        CONVERTERS: ("kenning.converters", ModelConverter),
    }


def get_all_subclasses(
    module_path: str,
    cls: Type,
    raise_exception: bool = False,
    import_classes: bool = True,
    show_warnings: bool = True,
    blacklist: Sequence[str] = (),
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
    import_classes : bool
        Whether to import classes into memory or just return a list of modules.
    show_warnings : bool
        Tells whether method should print warnings if modules could not be
        imported.
    blacklist : Sequence[str]
        Prevent specified class name to be present in subclasses.

    Returns
    -------
    Union[List[Type], List[Tuple[str, str]]]
        When importing classes: List of all final subclasses of given class.
        When not importing classes: list of tuples with name and module path
        of the class.

    Raises
    ------
    ModuleNotFoundError, ImportError
        When modules could not be imported.
    Exception
        If some unspecified errors occurred during imports.
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

        It is an internal function updating the `subclasses`, `checked_classes`
        structures.

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
            elif b.id in [x.__name__ for x in cls.__subclasses__()]:
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
        if subclass_name in blacklist:
            continue

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


def classes_from_full_dict_config(
    full_dict_config: KenningBlockConfigDict
) -> Dict[ConfigKey, Union[type, List[type]]]:
    """
    Loads and returns block classes based on a Kenning configuration dict.

    Parameters
    ----------
    full_dict_config: KenningBlockConfigDict
        Special Kenning block configuration dict, with format as specified in
        kenning.dispatcher.block_config

    Returns
    -------
    Dict[ConfigKey, Union[type, List[type]]]
        Classes. For each ConfigKey there is a single class, except for
        ConfigKey.optimizers (for which there is a list of classes).
    """
    keys = full_dict_config[BLOCK_CONFIGURATIONS_KEY].keys()

    keys = [ConfigKey(key) for key in keys]

    classes = {}

    for key in keys:
        if key == ConfigKey.optimizers:
            optimizer_classes = []
            for optimizer in list(
                full_dict_config[BLOCK_CONFIGURATIONS_KEY][
                    ConfigKey.optimizers
                ].keys()
            ):
                if optimizer:
                    optimizer_classes.append(
                        load_class_by_type(
                            optimizer, ConfigKey.optimizers.value
                        )
                    )
            classes[ConfigKey.optimizers] = optimizer_classes
        else:
            class_name = list(
                full_dict_config[BLOCK_CONFIGURATIONS_KEY][key].keys()
            )[0]
            if class_name:
                classes[key] = load_class_by_type(class_name, key.value)
    return classes


def objs_from_full_dict_config(
    full_dict_config: KenningBlockConfigDict
) -> Dict[ConfigKey, Any]:
    """
    Builds block objects based on a Kenning configuration dict.

    Parameters
    ----------
    full_dict_config: KenningBlockConfigDict
        Special Kenning block configuration dict, with format as specified in
        kenning.dispatcher.block_config

    Returns
    -------
    Dict[ConfigKey, Any]
        Parsed objects. For each ConfigKey there is a single object, except for
        ConfigKey.optimizers, for which there is a list of objects.
    """
    classes = classes_from_full_dict_config(full_dict_config)

    objs = {
        key: cls.build_from_config(full_dict_config)
        for key, cls in classes.items()
        if key
        in [
            ConfigKey.platform,
            ConfigKey.protocol,
            ConfigKey.dataset,
            ConfigKey.runtime,
            ConfigKey.runtime_builder,
        ]
    }

    dataset = objs.get(ConfigKey.dataset)

    if modelwrappercls := classes.get(ConfigKey.model_wrapper):
        objs[ConfigKey.model_wrapper] = (
            modelwrappercls.build_from_config(
                full_dict_config, dataset=dataset
            )
            if modelwrappercls
            else None
        )

    if reportcls := classes.get(ConfigKey.report):
        objs[ConfigKey.report] = (
            reportcls.build_from_config(
                full_dict_config,
                model_wrapper=objs[ConfigKey.model_wrapper]
                if ConfigKey.model_wrapper in objs
                else None,
            )
            if reportcls
            else None
        )

    # TODO: This is a temporal solution, in future dataconverter
    # should be parsed separately
    if model := objs.get(ConfigKey.model_wrapper):
        from kenning.dataconverters.modelwrapper_dataconverter import (
            ModelWrapperDataConverter,
        )

        objs[ConfigKey.dataconverter] = ModelWrapperDataConverter(model)

    if compilercls_list := classes.get(ConfigKey.optimizers):
        objs[ConfigKey.optimizers] = [
            compilercls.build_from_config(
                full_dict_config,
                dataset=dataset,
                model_wrapper=objs[ConfigKey.model_wrapper]
                if ConfigKey.model_wrapper in objs
                else None,
            )
            for compilercls in compilercls_list
        ]
    else:
        objs[ConfigKey.optimizers] = []

    automl_optional_deps = {}
    if ConfigKey.platform in objs:
        automl_optional_deps["platform"] = objs[ConfigKey.platform]
    if ConfigKey.optimizers in objs:
        automl_optional_deps["optimizers"] = objs[ConfigKey.optimizers]
    if ConfigKey.runtime in objs:
        automl_optional_deps["runtime"] = objs[ConfigKey.runtime]

    if automl := classes.get(ConfigKey.automl):
        objs[ConfigKey.automl] = (
            automl.build_from_config(
                full_dict_config, dataset=dataset, **automl_optional_deps
            )
            if automl
            else None
        )

    if inference_loop := classes.get(ConfigKey.inference_loop):
        objs[ConfigKey.inference_loop] = inference_loop.build_from_config(
            full_dict_config,
            dataset=objs.get(ConfigKey.dataset),
            dataconverter=objs.get(ConfigKey.dataconverter),
            model_wrapper=objs.get(ConfigKey.model_wrapper),
            platform=objs.get(ConfigKey.platform),
            protocol=objs.get(ConfigKey.protocol),
            runtime=objs.get(ConfigKey.runtime),
        )

    return objs


def parse_classes(
    classes: List[Type],
    args: argparse.Namespace,
    not_parsed: List[str],
    override_only: bool = False,
) -> argparse.Namespace:
    """
    Parses remaining arguments from class definitions determined by
    ``form_argparse`` into ``argparse.Namespace``.

    Parameters
    ----------
    classes: List[Type]
        Classes to load.
    args : argparse.Namespace
        Initial namespace.
    not_parsed : List[str]
        Remaining arguments.
    override_only : bool
        True if ``overridable`` parameters should be parsed.

    Returns
    -------
    argparse.Namespace
        Parsed class parameters.

    Raises
    ------
    ParserHelpException
        Raised when help is requested in arguments.
    argparse.ArgumentParser
        Raised when report types cannot be deduced from measurements data.
    """
    command = get_command(with_slash=False)

    parser = argparse.ArgumentParser(
        " ".join(map(lambda x: x.strip(), command)) + "\n",
        parents=[
            cls.form_argparse(args, override_only=override_only)[0]
            for cls in classes
        ],
        add_help=False,
    )

    if args.help:
        raise ParserHelpException(parser)

    args, not_parsed = parser.parse_known_args(not_parsed, namespace=args)

    if not_parsed:
        raise argparse.ArgumentError(
            None, f"unrecognized arguments: {' '.join(not_parsed)}"
        )

    return args


def load_class(module_path_or_name: str) -> Type:
    """
    Loads class given in the `module_path_or_name`.

    Parameters
    ----------
    module_path_or_name : str
        Either a module-like path to the class
        or the name of the class.

    Returns
    -------
    Type
        Loaded class.

    """
    if is_class_name(module_path_or_name):
        cls_name = module_path_or_name
        module_path = get_module_path(module_path_or_name)
    else:
        module_path, cls_name = module_path_or_name.rsplit(".", 1)

    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)
    return cls


def is_class_name(name: str) -> bool:
    """
    Check if `name` is a valid class name.

    It does not check if a class with the given class is implemented.
    Only validity of `name` is verified.

    Parameters
    ----------
    name : str
        Potential class name to be checked.

    Returns
    -------
    bool
        Whether `name` is a valid class name.
    """
    return "." not in name and "_" not in name


def get_module_path(class_name: str) -> str:
    """
    Get a path-like module for a provided class name.

    Parameters
    ----------
    class_name : str
        Name of the class.

    Returns
    -------
    str
        Path-like location of a Python module.

    Raises
    ------
    AmbiguousModuleError
        Raised if two or more classes named `class_name` exist.
    ModuleNotFoundError
        Raised if there is no class matching `class_name`.
    """
    matching_paths: List[str] = []
    for block_name in get_base_classes_dict().keys():
        cls = load_class_by_type(
            path=class_name, block_type=block_name, log_errors=False
        )
        if cls:
            matching_paths.append(cls.__module__)

    matching_paths_count = len(matching_paths)
    if matching_paths_count < 1:
        raise ModuleNotFoundError(
            f"None of the classes match {class_name!r}."
            "Check the class name for typos or provide a full module path."
        )
    if matching_paths_count > 1:
        raise ModuleNotFoundError(
            f"More than one class matches {class_name!r}."
            "Provide a full module path, instead."
        )
    [module_path] = matching_paths
    return module_path


def load_class_by_type(
    path: Optional[str],
    block_type: Optional[str] = None,
    log_errors: bool = True,
) -> Optional[Type]:
    """
    Loads the class based on its name and type, or using full path.

    Parameters
    ----------
    path : Optional[str]
        A path to the class or full class name (requires block_type).
    block_type : Optional[str]
        Type of Kenning block, i.e. "optimizers", "platforms". If specified
        then type in config does not require to specify full class path.
    log_errors : bool
        Whether errors should be logged. By default, True.
        Useful to turn off to prevent logging false positives
        when looking for a class across modules.

    Returns
    -------
    Optional[Type]
        Loaded class or None if class cannot be found.
    """
    if path is None:
        return None
    base_classes_dict = get_base_classes_dict()
    if (
        block_type is not None
        and block_type in base_classes_dict
        and "." not in path
    ):
        module_path, base_class = base_classes_dict[block_type]
        subclasses = get_all_subclasses(
            module_path, base_class, import_classes=False
        )

        cls_type = None
        for subcls_name, subcls_module_path in subclasses:
            if subcls_name == path:
                cls_type = f"{subcls_module_path}.{subcls_name}"
                break

        if cls_type is None and log_errors:
            KLogger.error(f"Could not find class of {path}")
    else:
        cls_type = path

    if cls_type is not None:
        module_path, class_name = cls_type.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    return None


@contextmanager
def append_to_sys_path(paths: List[Path]) -> Generator[None, None, None]:
    """
    Context manager extending `sys.path` with given directories.

    Parameters
    ----------
    paths : List[Path]
        The list with directories to extend `sys.path` with.

    Yields
    ------
    None
    """
    prev_sys_path = sys.path
    sys.path = list(map(str, paths)) + sys.path[:]

    try:
        yield
    finally:
        sys.path = prev_sys_path


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

    import kenning

    modulename = None
    if not str(Path(kenning.__file__).resolve()).endswith("kenning"):
        modulename = get_kenning_submodule_from_path(kenning.__file__)

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
