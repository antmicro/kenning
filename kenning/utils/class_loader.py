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
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from kenning.cli.parser import ParserHelpException
from kenning.core.automl import AutoML
from kenning.core.dataconverter import DataConverter
from kenning.core.dataprovider import DataProvider
from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.onnxconversion import ONNXConversion
from kenning.core.optimizer import Optimizer
from kenning.core.outputcollector import OutputCollector
from kenning.core.platform import Platform
from kenning.core.protocol import Protocol
from kenning.core.report import Report
from kenning.core.runner import Runner
from kenning.core.runtime import Runtime
from kenning.core.runtimebuilder import RuntimeBuilder
from kenning.utils.args_manager import (
    convert_to_jsontype,
    get_parsed_args_dict,
    to_namespace_name,
)
from kenning.utils.logger import KLogger

OPTIMIZERS = "optimizers"
RUNNERS = "runners"
DATA_PROVIDERS = "dataproviders"
DATA_CONVERTERS = "dataconverters"
DATASETS = "datasets"
MODEL_WRAPPERS = "modelwrappers"
ONNX_CONVERSIONS = "onnxconversions"
OUTPUT_COLLECTORS = "outputcollectors"
PLATFORMS = "platforms"
RUNTIME_BUILDERS = "runtimebuilders"
RUNTIME_PROTOCOLS = "protocols"
RUNTIMES = "runtimes"
AUTOML = "automl"
REPORT = "report"


class ConfigKey(str, Enum):
    """
    Enum with fields available in configuration.

    `name` property defines key in configuration,
    `value` property defines type of the class given field can store.
    """

    dataset = DATASETS
    runtime = RUNTIMES
    optimizers = OPTIMIZERS
    platform = PLATFORMS
    protocol = RUNTIME_PROTOCOLS
    model_wrapper = MODEL_WRAPPERS
    runtime_builder = RUNTIME_BUILDERS
    dataconverter = DATA_CONVERTERS
    automl = AUTOML
    report = REPORT


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
        ONNX_CONVERSIONS: ("kenning.onnxconverters", ONNXConversion),
        OUTPUT_COLLECTORS: ("kenning.outputcollectors", OutputCollector),
        PLATFORMS: ("kenning.platforms", Platform),
        RUNTIME_BUILDERS: ("kenning.runtimebuilders", RuntimeBuilder),
        RUNTIME_PROTOCOLS: ("kenning.protocols", Protocol),
        RUNTIMES: ("kenning.runtimes", Runtime),
        AUTOML: ("kenning.automl", AutoML),
        REPORT: ("kenning.report", Report),
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
    import_classes : bool
        Whether to import classes into memory or just return a list of modules.
    show_warnings : bool
        Tells whether method should print warnings if modules could not be
        imported.

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


def objs_from_json(
    json_cfg: Dict[str, Any],
    keys: Set[ConfigKey],
    override: Optional[Tuple[argparse.Namespace, List[str]]] = None,
) -> Dict[ConfigKey, Any]:
    """
    Loads the objects from configuration, specified by keys.

    Parameters
    ----------
    json_cfg : Dict[str, Any]
        A JSON object containing entire configuration, from which the class is
        retrieved.
    keys : Set[ConfigKey]
        Keys that correspond to classes of objects that should be loaded.
    override : Optional[Tuple[argparse.Namespace, List[str]]]
        If not none, arguments will be used to override config parameters.

    Returns
    -------
    Dict[ConfigKey, Any]
        Parsed parameters.
    """
    keys_regular = set(
        [
            ConfigKey.platform,
            ConfigKey.protocol,
            ConfigKey.dataset,
            ConfigKey.runtime,
            ConfigKey.runtime_builder,
            ConfigKey.report,
        ]
    ).intersection(keys)

    if override:
        args, not_parsed = override
        merge_argparse_and_json(keys_regular, json_cfg, args, not_parsed)

    objs = {key: obj_from_json(json_cfg, key) for key in keys_regular}

    dataset = objs.get(ConfigKey.dataset)

    if ConfigKey.model_wrapper in keys:
        objs[ConfigKey.model_wrapper] = obj_from_json(
            json_cfg, ConfigKey.model_wrapper, dataset=dataset
        )

    model_wrapper = objs.get(ConfigKey.model_wrapper)

    if ConfigKey.dataconverter in keys:
        objs[ConfigKey.dataconverter] = any_from_json(
            json_cfg.get(ConfigKey.runtime.name, {}).get("data_converted", {}),
            block_type="dataconverters",
        )

    if ConfigKey.optimizers in keys:
        objs[ConfigKey.optimizers] = [
            any_from_json(
                optimizer_cfg,
                block_type=ConfigKey.optimizers.value,
                dataset=dataset,
                model_wrapper=model_wrapper,
            )
            for optimizer_cfg in json_cfg.get(
                ConfigKey.optimizers.name,
                [],
            )
        ]
    else:
        objs[ConfigKey.optimizers] = []

    return objs


def merge_argparse_and_json(
    keys: Set[ConfigKey],
    json_cfg: Dict[str, Any],
    args: argparse.Namespace,
    not_parsed: List[str],
):
    """
    Update ``json_cfg`` with overridable values from not parsed arguments.

    Parameters
    ----------
    keys : Set[ConfigKey]
        Keys that correspond to classes of objects that should be overridden.
    json_cfg : Dict[str, Any]
        A JSON object containing entire configuration, from which the class is
        retrieved.
    args : argparse.Namespace
        Initial namespace.
    not_parsed : List[str]
        Remaining arguments.
    """
    keys = keys.difference([ConfigKey.dataconverter, ConfigKey.optimizers])
    classes = {
        key: cls for key in keys if (cls := load_class_by_key(json_cfg, key))
    }
    args = parse_classes(
        list(classes.values()), args, not_parsed, override_only=True
    )

    for key, cls in classes.items():
        if key.name in json_cfg:
            if params := get_parsed_args_dict(cls, args, override_only=True):
                json_cfg[key.name]["parameters"] = dict(
                    json_cfg[key.name].get("parameters", {}),
                    **convert_to_jsontype(params),
                )


def objs_from_argparse(
    args: argparse.Namespace,
    not_parsed: List[str],
    keys: Set[ConfigKey],
    required: Optional[Callable[[Dict[ConfigKey, Any]], Any]] = None,
) -> Dict[ConfigKey, Any]:
    """
    Parses objects from arguments, specified by keys.

    Parameters
    ----------
    args : argparse.Namespace
        Initial namespace.
    not_parsed : List[str]
        Remaining arguments.
    keys : Set[ConfigKey]
        Keys that correspond to classes of objects that should be loaded.
    required : Optional[Callable[[Dict[ConfigKey, Any]], Any]]
        Callback that verifies used classes.

    Returns
    -------
    Dict[ConfigKey, Any]
        Parsed objects.
    """
    classes = {
        key: cls
        for key in keys
        if (class_arg := getattr(args, to_namespace_name(key), None))
        if (cls := load_class(class_arg))
    }

    if required:
        required(classes)

    KLogger.debug(f"Classes: {classes}")

    args = parse_classes(list(classes.values()), args, not_parsed)

    objs = {
        key: cls.from_argparse(args)
        for key, cls in classes.items()
        if key
        in [
            ConfigKey.platform,
            ConfigKey.protocol,
            ConfigKey.dataset,
            ConfigKey.runtime,
            ConfigKey.report,
        ]
    }

    dataset = objs.get(ConfigKey.dataset)

    if modelwrappercls := classes.get(ConfigKey.model_wrapper):
        objs[ConfigKey.model_wrapper] = (
            modelwrappercls.from_argparse(dataset, args)
            if modelwrappercls
            else None
        )

    # TODO: This is a temporal solution, in future dataconverter
    # should be parsed separately
    if model := objs.get(ConfigKey.model_wrapper):
        from kenning.dataconverters.modelwrapper_dataconverter import (
            ModelWrapperDataConverter,
        )

        objs[ConfigKey.dataconverter] = ModelWrapperDataConverter(model)

    if compilercls := classes.get(ConfigKey.optimizers):
        objs[ConfigKey.optimizers] = [compilercls.from_argparse(dataset, args)]
    else:
        objs[ConfigKey.optimizers] = []

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
    KLogger.debug(f"Command: {command}")

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


def obj_from_json(
    json_cfg: Dict[str, Any], key: ConfigKey, **kwargs
) -> Optional[Any]:
    """
    Loads the object from configuration, specified by key.

    Parameters
    ----------
    json_cfg : Dict[str, Any]
        A JSON object containing entire configuration, from which the field is
        retrieved and converted into an object.
    key : ConfigKey
        Chooses the field from configuration and class type.
    **kwargs :
        Additional arguments

    Returns
    -------
    Optional[Any]
        If a class is available and contains `from_json` method, it
        returns object of this class.
    """
    return any_from_json(json_cfg.get(key.name, {}), key.value, **kwargs)


def any_from_json(
    json_cfg: Dict[str, Any], block_type: Optional[str] = None, **kwargs
) -> Optional[Any]:
    """
    Loads the object using `from_json` method, if available.

    Parameters
    ----------
    json_cfg : Dict[str, Any]
        A JSON object snippet with `type` parameter, specifying the
        full name of the class, and `parameters` parameter, with list
        of constructor arguments for the class.
    block_type : Optional[str]
        Type of Kenning block, i.e. "optimizers", "platforms". If specified
        then type in config does not require to specify full class path.
    **kwargs :
        Additional arguments

    Returns
    -------
    Optional[Any]
        If a class is available and contains `from_json` method, it
        returns object of this class.
    """
    if block_type == ConfigKey.report.value and "type" not in json_cfg:
        json_cfg["type"] = "kenning.report.markdown_report.MarkdownReport"

    if cls := load_class_from_json(json_cfg, block_type):
        return cls.from_json(json_cfg.get("parameters", {}), **kwargs)


def load_class_from_json(
    json_cfg: Dict[str, Any], block_type: Optional[str] = None
) -> Optional[Type]:
    """
    Loads class from configuration if it exists `from_json` method is
    available.

    Parameters
    ----------
    json_cfg : Dict[str, Any]
        A JSON object snippet with `type` parameter, specifying the
        full name of the class, and `parameters` parameter, with list
        of constructor arguments for the class.
    block_type : Optional[str]
        Type of Kenning block, i.e. "optimizers", "platforms". If specified
        then type in config does not require to specify full class path.

    Returns
    -------
    Optional[Type]
        A class if it is available and contains `from_json` method.
    """
    if "type" not in json_cfg:
        return None

    cls = load_class_by_type(json_cfg["type"], block_type)
    if cls is None or not hasattr(cls, "from_json"):
        return None
    return cls


def load_class_by_key(
    json_cfg: Dict[str, Any], key: ConfigKey
) -> Optional[Type]:
    """
    Loads the object from configuration, specified by key.

    Parameters
    ----------
    json_cfg : Dict[str, Any]
        A JSON object containing entire configuration, from which the class is
        retrieved.
    key : ConfigKey
        Chooses the class from configuration.

    Returns
    -------
    Optional[Type]
        A class if it is available and contains `from_json` method.
    """
    return load_class_from_json(json_cfg.get(key.name, {}), key.value)


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

    KLogger.debug(f"Paths added to sys.path: {paths}")

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
