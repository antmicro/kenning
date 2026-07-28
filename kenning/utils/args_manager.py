# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for preparing and serializing class arguments.
"""
from __future__ import annotations

import argparse
import json
import os.path
import typing
from abc import ABC
from pathlib import Path
from types import UnionType
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    GenericAlias,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
    get_origin,
)

import jsonschema
import numpy as np
from typeguard import check_type

from kenning.cli.parser import get_used_subcommands
from kenning.core.exceptions import (
    ArgsManagerConvertError,
    ConfigurationError,
    InvalidArgumentsError,
)
from kenning.dispatcher.block_config import (
    BLOCK_CONFIG_PARAMETERS_KEY,
    BLOCK_CONFIGURATIONS_KEY,
    BLOCK_DIRECT_ARGUMENTS_KEY,
    UNAFFILIATED_PARAMETERS_KEY,
    ConfigKey,
    KenningBlockConfigDict,
    from_flag_to_name,
)
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import ResourceURI

"""
arguments_structure is a mapping (argument_name -> keywords)

Supported keywords:
argparse_name: Name that is prompted as an argparse argument.
    If argparse_name is not specified then JSON uses argument_name
    and argparse adds -- prefix and change all underscores to hyphens.
    E.g. model_path -> --model-path.

    If it is specified then argparse uses it and JSON uses argument with
    hyphens stripped from the prefix and changed to underscores.
    E.g. --some-name -> some_name.
description: Description of the argument.
type: Same as 'type' in argparse. The argument is converted to this value.
    Possible values for type: [int, float, str, bool, Path, list, object].

    Note that for bool types, the argument has to declare its default value.
default: Default value of the argument.
required: Determines whether the argument is required.
enum: List of possible values of the argument.
nullable: Determines whether the argument can be None.
    Possible values for nullable: [True, False].
    By default it is False.
    It is used only for JSON.
    Note that for a property to be nullable it has to be either a list
    or have a type defined.
AutoML: Bool specifying whether parameter is used by AutoML.
list_range: Tuple with lower and upper bound of list length.
    Available only if `type` is list.
item_range: Tuple with lower and upper bound of value or list's elements.
    Available only if `type` is int, float or list.

Examples:

'compiled_model_path': {
    'argparse_name': '--model-path',
    'description': 'The path to the compiled model output',
    'type': Path,
    'required': True,
    'enum': ['/home/Documents', '/tmp']
}

'inputdims': {
    'description': 'Dimensionality of the inputs',
    'type': list[int],
    'default': [224, 224, 3],
    'nullable': True
}

'encoder_neuron_list': {
    'description': 'List of dense layer dimensions of encoder',
    'type': list[int],
    'default': [16, 8],
    'AutoML': True,
    'list_range': (2, 6),
    'item_range': (4, 48),
}
"""

supported_keywords = [
    "argparse_name",
    "description",
    "type",
    "default",
    "required",
    "enum",
    "nullable",
    "subcommands",  # DEPRECATED
    "overridable",
    # optional keys (do not need to be in the argument definition and have no
    # default values).
    "AutoML",
    "list_range",
    "item_range",
]


def from_name_to_flag(s: str) -> str:
    """
    Converts entry from arguments_structure to argparse entry.
    """
    return "--" + s.replace("_", "-")


def to_namespace_name(s: ConfigKey) -> str:
    """
    Converts class key to the name that is stored in the
    ``argparse.Namespace``.

    Parameters
    ----------
    s : ConfigKey
        Configuration key of the class.

    Returns
    -------
    str
        Name of the attribute in the namespace.
    """
    from kenning.utils.class_loader import ConfigKey

    if s == ConfigKey.optimizers:
        return "compiler_cls"
    return s.name.replace("_", "") + "_cls"


def convert_to_jsontype(v: Any) -> Any:
    """
    Converts entry to JSON-like format.

    Parameters
    ----------
    v: Any
        Input entry

    Returns
    -------
    Any
        Converted entry to JSON-like format
    """
    if isinstance(v, list):
        return [convert_to_jsontype(e) for e in v]
    if isinstance(v, dict):
        return {key: convert_to_jsontype(e) for key, e in v.items()}
    if isinstance(v, (ResourceURI)):
        return v.origin
    if isinstance(v, (Path)):
        return str(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def ensure_exclusive_cfg_or_flags(
    args: argparse.Namespace,
    flag_config_names: Sequence[str],
    required: Optional[Iterable[int]] = None,
):
    """
    Verifies exclusion of file-based or flag-based configuration.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    flag_config_names : Sequence[str]
        Flags supported by the command.
    required : Optional[Iterable[int]]
        If provided, only selected flags are checked to not be missing.

    Raises
    ------
    ArgumentError
        Raised when verification fails.
    """
    flag_config_not_none = [
        getattr(args, name, None) is not None for name in flag_config_names
    ]
    if args.json_cfg is None:
        # Exclusion violated
        if not any(flag_config_not_none):
            raise argparse.ArgumentError(
                None, "JSON or flag config is required."
            )

        # Missing arguments
        missing_args = [
            f"'{flag_config_names[i]}'"
            for i in required or range(len(flag_config_names))
            if not flag_config_not_none[i]
        ]
        if missing_args and not args.help:
            report_missing(missing_args)

    if args.json_cfg is not None and any(flag_config_not_none):
        raise argparse.ArgumentError(
            None,
            "JSON and flag configurations are mutually exclusive. "
            "Please use only one method of configuration.",
        )


def report_missing(names: List[str]):
    """
    Reports missing arguments given their names.

    Parameters
    ----------
    names : List[str]
        Names of the arguments to report.

    Raises
    ------
    argparse.ArgumentError
        Always raised.
    """
    raise argparse.ArgumentError(
        None, f"missing required arguments: {', '.join(names)}"
    )


type_to_jsontype = {
    Path: "string",
    ResourceURI: "string",
    str: "string",
    float: "number",
    int: "integer",
    bool: "boolean",
    object: "object",
    list: "array",
}

jsontype_to_type = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
}

# Dictionary with custom converter methods
converter_override = {
    object: dict,
}


def convert(converters: Iterable[Callable], v: Any) -> Any:
    """
    Converts value using given converter
    or raises exception if it is not possible.

    Parameters
    ----------
    converters : Iterable[Callable]
        List of convertes.
    v : Any
        Value to be converted.

    Returns
    -------
    Any
        Converted value.

    Raises
    ------
    ArgsManagerConvertError
        If value cannot be converted.
    """
    c_value = None
    for converter in converters:
        try:
            c_value = converter_override.get(converter, converter)(v)
        except Exception:
            KLogger.warn(
                f"Cannot convert value {v} with converter {converter}"
            )
        else:
            break
    if c_value is None:
        raise ArgsManagerConvertError(
            f"Value {v} cannot be converted by {converters}"
        )
    return c_value


def try_to_load_param_from_ros(param_name: str) -> Any:
    """
    A function that will try to gather parameters
    from ROS 2 params gathered by global ROS 2 Node.

    Parameters
    ----------
    param_name : str
        Name of the desired parameter.

    Returns
    -------
    Any
        A parameter with name param_name.
    """
    KLogger.debug(f"Trying to get values from ROS2 {param_name}")
    try:
        from kenning.utils.ros2_global_context import ROS2GlobalContext

        if ROS2GlobalContext.node is None:
            return None

        param = ROS2GlobalContext.get_param(param_name)

        if param is not None:
            KLogger.debug(
                f"Got {param_name} parameter from ROS 2 with value: {param}"
            )

        return param

    except ModuleNotFoundError:
        return None


def get_parsed_json_dict(schema: Dict, json_dict: Dict) -> Dict:
    """
    Validates the given dictionary with the schema.
    Then it adds default values for missing
    arguments and converts the arguments to the appropriate types
    as jsonschema can not do that. Finally it provides new names
    that match the constructor arguments.

    Parameters
    ----------
    schema : Dict
        Schema to validate with.
    json_dict : Dict
        Dictionary to validate.

    Returns
    -------
    Dict
        Validated dictionary with arguments.
    """
    jsonschema.validate(instance=json_dict, schema=schema)

    converted_json_dict = {}

    # Including default values
    for name, keywords in schema["properties"].items():
        if (value := try_to_load_param_from_ros(name)) is not None:
            converted_json_dict[name] = value
        elif name not in json_dict and "default" in keywords:
            converted_json_dict[name] = keywords["default"]
        elif name in json_dict:
            converted_json_dict[name] = json_dict[name]

    # Converting arguments to the final types.
    # Following argparse convention, if our value is None
    # then we do not convert it.
    for name, value in converted_json_dict.items():
        keywords = schema["properties"][name]

        if "convert-type" not in keywords or not value:
            converted_json_dict[name] = value
            continue

        # do not convert to an object as this should be done by the class
        # which called this function
        if keywords["convert-type"] is object:
            converted_json_dict[name] = value
            continue

        converter = keywords["convert-type"]
        if not isinstance(converter, (list, tuple)):
            converter = [converter]

        if (
            "type" in keywords
            and "array" in keywords["type"]
            and isinstance(value, (list, tuple))
        ):
            converted_json_dict[name] = [
                convert(converter, v) if v else v for v in value
            ]
        else:
            converted_json_dict[name] = convert(converter, value)

    # Changing names so they match the constructor arguments
    converted_json_dict = {
        schema["properties"][name]["real_name"]: value
        for name, value in converted_json_dict.items()
    }

    return converted_json_dict


def get_parsed_args_dict(
    cls: type, args: argparse.Namespace, override_only: bool = False
) -> Dict:
    """
    Converts namespace provided by arguments parser into dictionary.

    Parameters
    ----------
    cls : type
        Class of object being parsed.
    args : argparse.Namespace
        Namespace provided by arguments parser.
    override_only : bool
        True if only parameters marked as `overridable` should be parsed.

    Returns
    -------
    Dict
        Dictionary with arguments.

    Raises
    ------
    Exception:
        Raised when default values for arguments are not specified
    """
    # retrieve all arguments from arguments_structure of this class and all of
    # its parent classes
    args_structure = {}
    for curr_cls in traverse_parents_with_args(cls):
        args_structure = dict(
            args_structure,
            **{
                name: arg
                for name, arg in curr_cls.arguments_structure.items()
                if not override_only or arg.get("overridable", True)
            },
        )

    # parse arguments
    parsed_args = {}
    for arg_name, arg_properties in args_structure.items():
        if "argparse_name" in arg_properties:
            argparse_name = from_flag_to_name(arg_properties["argparse_name"])
        else:
            argparse_name = arg_name

        value = try_to_load_param_from_ros(argparse_name)

        if value is None and hasattr(args, argparse_name):
            value = getattr(args, argparse_name)

        if value is None and not override_only:
            if "default" in arg_properties.keys():
                value = arg_properties["default"]
            elif (
                "required" in arg_properties.keys()
                and arg_properties["required"]
            ):
                raise Exception(
                    f"No default value provided for {argparse_name}"
                )
            else:
                value = None

        if value is None and override_only:
            continue

        # For arguments of type 'object' value is embedded in a JSON file
        if "type" in arg_properties and arg_properties["type"] is object:
            if value is None:
                try:
                    value = arg_properties["default"]
                    parsed_args[arg_name] = value
                    continue
                except KeyError:
                    raise Exception(
                        f"No default value provided for {argparse_name}"
                    )

            if not os.path.exists(value):
                raise Exception(
                    f"JSON configuration file {value} doesnt exist"
                )

            with open(value, "r") as file:
                value = json.load(file)
                parsed_args[arg_name] = value
                continue

        # convert type
        if (
            "type" in arg_properties
            and value is not None
            and get_origin(arg_properties["type"]) not in (UnionType, Union)
        ):
            _type = arg_properties["type"]

            origin = get_origin(_type)

            if origin is not None:
                _type = origin

            KLogger.debug(f"Converting value: '{value}' with {_type}")

            value = _type(value)

        parsed_args[arg_name] = value

    return parsed_args


def get_type(
    _type: Union[type, GenericAlias],
) -> Tuple[type, Optional[List[Union[type, Tuple[type, ...]]]]]:
    """
    Returns proper type and its optional sub types.

    Parameters
    ----------
    _type : Union[type, GenericAlias]
        Type like str, int or list[int | float, str].

    Returns
    -------
    Tuple[type, Optional[List[Union[type, Tuple[type, ...]]]]]
        * The main type,
        * The list of subtypes of e.g. list,
        where union types are gathered in tuple. Currently, do not
        support nested types.
    """
    main_type = get_origin(_type)

    if main_type is None:
        return _type, None

    sub_types = [
        arg.__args__ if isinstance(arg, UnionType) else arg
        for arg in _type.__args__
    ]
    return main_type, sub_types


def traverse_parents_with_args(cls: type) -> Generator[type, None, None]:
    """
    Traverses parents of a given class that have ``arguments_structure``
    defined.

    Note that traversal is executed with BFS strategy, not MRO.

    Parameters
    ----------
    cls : type
        Class to get parents from.

    Yields
    ------
    type
        Parent class.
    """
    classes = [cls]
    while len(classes):
        curr_cls = classes.pop(0)
        classes.extend(curr_cls.__bases__)
        if not hasattr(curr_cls, "arguments_structure"):
            continue
        yield curr_cls


def add_argparse_argument(
    group: argparse._ArgumentGroup,
    struct: Dict[str, Dict],
    args: argparse.Namespace,
    *names: str,
    override_only: bool = False,
):
    """
    Adds arguments to the argparse group based on the given
    struct and names. If no names are given it uses all
    properties of the struct.

    Note that the function modifies the given group.

    If argument with name 'argparse_name' already exists in the
    group the existing argument is overridden with the new one.

    Parameters
    ----------
    group : argparse._ArgumentGroup
        Argparse group that is filled with new properties.
    struct : Dict[str, Dict]
        Struct with properties described.
    args : argparse.Namespace
        Arguments from ArgumentParser object.
    *names : str
        Names of the properties that are to be added to the group.
        If empty every property in struct is used.
    override_only : bool
        True if parameter set in file configuration can be overridden from
        `argparse`.

    Raises
    ------
    KeyError :
        Raised if there is a keyword that is not recognized.
    """
    from kenning.cli.config import AVAILABLE_COMMANDS

    if not names:
        names = struct.keys()

    for name in names:
        prop = struct[name]

        for p in prop:
            if p not in supported_keywords:
                raise KeyError(f"{p} is not a supported keyword")

        if "subcommands" in prop:
            required_subcommands = prop["subcommands"] or AVAILABLE_COMMANDS
            used_subcommands = get_used_subcommands(args)
            if not set(used_subcommands).intersection(required_subcommands):
                continue

        if "argparse_name" in prop:
            argparse_name = prop["argparse_name"]
        else:
            argparse_name = from_name_to_flag(name)

        keywords = {}
        if "type" in prop:
            prop_type, prop_sub_types = get_type(prop["type"])

            if prop_type in {UnionType, Union}:
                union_types = set(prop_sub_types)

                con_types = set([])

                for sub_type in union_types:
                    if get_origin(sub_type) in {list, tuple}:
                        type_args = get_args(sub_type)

                        con_types.update(type_args)
                    else:
                        con_types.add(sub_type)

                keywords["type"] = lambda v: convert(v=v, converters=con_types)

                if bool in union_types:
                    if not override_only:
                        keywords["default"] = False
                    keywords["const"] = True
                    keywords["nargs"] = "?"

                if any([get_type(x)[0] is list for x in union_types]):
                    keywords["nargs"] = "*"

            elif prop_type is bool:
                if not (
                    "default" in prop and prop["default"] in [True, False]
                ):
                    raise InvalidArgumentsError(
                        f"Default value should be defined for prop \
                            {name} and has value True or False."
                    )

                if override_only:
                    keywords["action"] = argparse.BooleanOptionalAction
                elif prop["default"]:
                    keywords["action"] = "store_false"
                else:
                    keywords["action"] = "store_true"
            elif prop_type is list:
                keywords["nargs"] = "+"
                converters = set()
                for sub_type in prop_sub_types:
                    if isinstance(sub_type, (list, tuple)):
                        converters.update(sub_type)
                    else:
                        converters.add(sub_type)

                def conversion(x, conv_list=converters):
                    return convert(v=x, converters=conv_list)

                keywords["type"] = conversion
            else:
                keywords["type"] = prop_type
        if "description" in prop:
            keywords["help"] = prop["description"]
        keywords["default"] = argparse.SUPPRESS
        keywords["required"] = False
        if "enum" in prop:
            keywords["choices"] = prop["enum"]

        group.add_argument(argparse_name, **keywords)


def add_parameterschema_argument(
    schema: Dict, struct: Dict[str, Dict], *names: str
):
    """
    Adds arguments to the schema based on the given
    struct and names. If no names are given it uses all
    properties of the struct.

    Note that the function modifies the given schema.

    If argument with name 'argschema_name' already exists in the
    schema, the argument will be skipped.

    Parameters
    ----------
    schema : Dict
        Schema that is filled with new properties.
    struct : Dict[str, Dict]
        Struct with properties described.
    *names : str
        Names of the properties that are to be added to the group.
        If empty every property in struct is used.

    Raises
    ------
    KeyError :
        Raised if there is a keyword that is not recognized or if there is
        already a property with a different `argparse_name` and the same
        property name.
    """
    if "properties" not in schema:
        schema["properties"] = {}

    if not names:
        names = struct.keys()

    for name in names:
        prop = struct[name]

        for p in prop:
            if p not in supported_keywords:
                raise KeyError(f"{p} is not a supported keyword")

        if "argparse_name" in prop:
            argparse_name = prop["argparse_name"]
            argschema_name = from_flag_to_name(argparse_name)
        else:
            argschema_name = name

        # Check if there is a property that is not going to be overridden
        # by the new property but has the same property name
        for k, p in schema["properties"].items():
            if p["real_name"] == name and k != argschema_name:
                raise KeyError(f"{p} already has a property name: {name}")

        # Continue if parameter already present in schema
        if argschema_name in schema["properties"]:
            continue

        schema["properties"][argschema_name] = {}
        keywords = schema["properties"][argschema_name]
        keywords["real_name"] = name

        # Set type based on provided Python type
        if "type" in prop:
            prop_type, prop_sub_type = get_type(prop["type"])

            if prop_type in {UnionType, Union}:
                union_types = set(prop_sub_type)

                con_types = set()

                types = []

                for sub_type in union_types:
                    origin = get_origin(sub_type)

                    if origin is None:
                        origin = sub_type

                    if origin in {list, tuple}:
                        type_args = get_args(sub_type)

                        con_types.update(type_args)

                    else:
                        con_types.add(sub_type)

                    types.append(type_to_jsontype[origin])

                keywords["convert-type"] = [*con_types]

                keywords["type"] = types

            elif prop_type is list and prop_sub_type:
                keywords["convert-type"] = prop_sub_type[0]
                keywords["items"] = {
                    "type": [
                        type_to_jsontype[t]
                        for t in (
                            prop_sub_type[0]
                            if isinstance(prop_sub_type[0], (list, tuple))
                            else [prop_sub_type[0]]
                        )
                    ],
                }
                if "enum" in prop:
                    keywords["items"]["enum"] = prop["enum"]

                keywords["type"] = [type_to_jsontype[prop_type]]
            else:
                keywords["convert-type"] = prop_type
                keywords["type"] = [type_to_jsontype[prop_type]]

        if "description" in prop:
            keywords["description"] = prop["description"]
        if "default" in prop:
            keywords["default"] = prop["default"]
        if "required" in prop and prop["required"]:
            if "required" not in schema:
                schema["required"] = []
            schema["required"].append(argschema_name)
        if "enum" in prop and (
            "type" not in keywords.keys() or "array" not in keywords["type"]
        ):
            keywords["enum"] = prop["enum"]
        if "nullable" in prop and prop["nullable"]:
            if "type" in keywords:
                keywords["type"] += ["null"]


class ArgumentsHandler(ABC):
    """
    Class responsible for creating parsers for arguments from command line or
    json configs.

    The child class should define its own `arguments_structure` and
    from_argparse/from_json methods so that it could be instantiated from
    command line arguments or json config.
    """

    arguments_structure = {}

    @classmethod
    def form_parameterschema(cls) -> Dict:
        """
        Creates parameter schema based on `arguments_structure` of class and
        its all parent classes.

        Returns
        -------
        Dict
            Parameter schema for the class.
        """
        parameterschema = {"type": "object", "additionalProperties": False}

        for curr_cls in traverse_parents_with_args(cls):
            add_parameterschema_argument(
                parameterschema, curr_cls.arguments_structure
            )

        return parameterschema

    @classmethod
    def get_full_arguments_structure(cls) -> Dict:
        """
        Takes 'arguments_structure' (dicts describing user-defined block
        parameters) from this class, and from all parent classes in the
        inheritance chain, and combines them. Settings from child always
        override settings from parent.

        Returns
        -------
        Dict
            Combined 'arguments_structure' from the inheritance chain.
        """
        full_arguments_structure = cls.arguments_structure
        for curr_cls in traverse_parents_with_args(cls):
            full_arguments_structure = (
                curr_cls.arguments_structure | full_arguments_structure
            )
        return full_arguments_structure

    @classmethod
    def form_argparse(
        cls, args: argparse.Namespace, override_only: bool = False
    ) -> Tuple[argparse.ArgumentParser, Optional[argparse._ArgumentGroup]]:
        """
        Creates argparse parser based on `arguments_structure` of class and its
        all parent classes.

        Parameters
        ----------
        args : argparse.Namespace
            Arguments from ArgumentParser object.
        override_only : bool
            True if only parameters marked as `overridable` should be parsed.

        Returns
        -------
        Tuple[argparse.ArgumentParser, Optional[argparse._ArgumentGroup]]
            Tuple with the argument parser object that can act as parent for
            program's argument parser, and the corresponding arguments' group
            pointer.
        """
        parser = argparse.ArgumentParser(
            add_help=False, conflict_handler="resolve"
        )
        group = None

        for curr_cls in traverse_parents_with_args(cls):
            add_argparse_argument(
                group=parser.add_argument_group(
                    title=f"{curr_cls.__name__} arguments"
                ),
                struct={
                    name: arg
                    for name, arg in curr_cls.arguments_structure.items()
                    if not override_only or arg.get("overridable", True)
                },
                args=args,
                override_only=override_only,
            )

        return parser, group

    @classmethod
    def from_argparse(
        cls, args: argparse.Namespace, **kwargs: Dict[str, Any]
    ) -> Any:
        """
        Constructor wrapper that takes the parameters from argparse args.

        Parameters
        ----------
        args : argparse.Namespace
            Arguments from ArgumentParser object.
        **kwargs : Dict[str, Any]
            Additional class-dependent arguments.

        Returns
        -------
        Any
            Instance created from provided args.
        """
        parsed_args_dict = get_parsed_args_dict(cls, args)

        return cls(**kwargs, **parsed_args_dict)

    @classmethod
    def from_json(cls, json_dict: Dict, **kwargs: Dict[str, Any]) -> Any:
        """
        Constructor wrapper that takes the parameters from json dict.

        This function checks if the given dictionary is valid according to the
        ``arguments_structure`` defined. If it is then it invokes the
        constructor.

        Parameters
        ----------
        json_dict : Dict
            Arguments for the constructor.
        **kwargs : Dict[str, Any]
            Additional class-dependent arguments.

        Returns
        -------
        Any
            Instance created from provided JSON.
        """
        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)

        cls_args = dict(parsed_json_dict, **kwargs)

        return cls(**cls_args)

    @classmethod
    def cast_complex_type(
        cls, value: Any, target_type: type | Tuple[type]
    ) -> Any:
        """
        Casts a value, or a does a recursive cast, if the value is a container.

        Parameters
        ----------
        value: Any
            Value to be converted. Can be any object, or a list, union or tuple
            of objects (also recursively). For example types such as:
            Tuple[List[List[Tuple[int]]], Tuple[str]] are supported.
        target_type: type | Tuple[type]
            A type or multiple types (in case of value being a container), that
            the value should be converted to. When type is a container type,
            and value is another container type - the kind of container will be
            preserved (this function cannot change container type, it only
            casts values inside containers).

        Returns
        -------
        Any
            Converted value.

        Raises
        ------
        ValueError
            Casting is not possible.
        """
        if isinstance(value, (list, List, Tuple, tuple)):
            if not hasattr(target_type, "__args__"):
                raise ValueError(
                    f"{value} is a container, but {target_type} is not a"
                    " container type."
                )
            subtypes = target_type.__args__
            # If length of the type tuple matches the length of container,
            # we may conclude that each container item has a different
            # corresponding type. Otherwise we assume that the entire type
            # applies to every item separately (e.g. List[str] vs
            # Tuple[int, str])
            return type(value)(
                [
                    cls.cast_complex_type(
                        value[i],
                        subtypes[i]
                        if len(subtypes) == len(value)
                        else subtypes,
                    )
                    for i in range(len(value))
                ]
            )
        if type(target_type) is tuple:
            target_type = target_type[0]
        if type(value) is target_type:
            return value
        try:
            check_type(value, value, target_type)
            return value
        except TypeError:
            ...
        try:
            return target_type(value)
        except TypeError:
            raise ValueError(f"Casting {value} to {target_type} failed.")

    @classmethod
    def convert_value(cls, value: Any, spec: Dict) -> Any:
        """
        Converts given value to type required by a given 'arguments_structure'
        argument.

        Parameters
        ----------
        value: Any
            Value to be converted. Can be any object, or a list, union or tuple
            of objects (also recursively). For example types such as:
            Union[List[List[Tuple[int]]], Tuple[str]] are supported, so are
            enums, None values etc.
        spec: Dict
            A value of the 'arguments_structure' dict, for the argument to be
            matched.

        Returns
        -------
        Any
            Converted value.

        Raises
        ------
        ArgsManagerConvertError
            Thrown when value cannot be safely converted.
        """
        if value is None:
            if spec["nullable"]:
                return None
            else:
                raise ArgsManagerConvertError(
                    f"{spec['type']} is not nullable."
                )
        if spec["enum"]:

            def check_list(value, spec):
                for element in value:
                    if element not in spec["enum"]:
                        raise ArgsManagerConvertError(
                            f"{element} not in {spec['enum']}"
                        )

            try:
                value = cls.cast_complex_type(value, spec["type"])
                if value in spec["enum"]:
                    return value
                if type(value) is not list:
                    raise ArgsManagerConvertError(
                        f"{value} not in {spec['enum']}"
                    )
                check_list(value, spec)
                return value
            except ValueError:
                ...
            try:
                value = cls.cast_complex_type(value, List[spec["type"]])
                check_list(value, spec)
                return value
            except ValueError:
                ...
            raise ArgsManagerConvertError(
                f"Enum value {value} does not match {spec['type']}."
            )
        if typing.get_origin(spec["type"]) in [Union, UnionType]:
            subtypes = list(spec["type"].__args__)
            for subtype in subtypes:
                try:
                    return cls.cast_complex_type(value, subtype)
                except ValueError:
                    ...
            raise ArgsManagerConvertError(
                f"Value {value} does not any of the types in the Union:"
                f" {spec['type']}."
            )
        try:
            return cls.cast_complex_type(value, spec["type"])
        except ValueError:
            raise ArgsManagerConvertError(
                f"Value {value} does not match {spec['type']}."
            )

    @classmethod
    def verify_type(cls, value: Any, spec: Dict) -> bool:
        """
        Checks, if the given value is of type specified by the argument spec,
        or can be cast to it.

        Parameters
        ----------
        value: Any
            Value to be checked. Can be any object, or a list, union or tuple
            of objects (also recursively).
        spec: Dict
            A value of the 'arguments_structure' dict, for the argument to be
            matched.

        Returns
        -------
        bool
            True if the value matches the argument. False otherwise.
        """

        def check_list_constraints(value, spec):
            if type(value) is not list:
                return True
            if "list_range" in spec and (
                spec["list_range"][0] > len(value)
                or spec["list_range"][1] < len(value)
            ):
                return False
            if "item_range" in spec:
                for subvalue in value:
                    try:
                        subvalue = float(subvalue)
                        if (
                            spec["item_range"][0] > subvalue
                            or spec["item_range"][1] < subvalue
                        ):
                            return False
                    except ValueError:
                        continue
            return True

        if value is None and spec["nullable"]:
            return True
        try:
            cls.convert_value(value, spec)
            return check_list_constraints(value, spec)
        except ArgsManagerConvertError:
            ...
        try:
            check_type(value, value, spec["type"])
            return check_list_constraints(value, spec)
        except TypeError:
            ...
        return False

    @classmethod
    def process_argument_spec(
        cls, argument_name: str, argument_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fills missing values in the given 'arguments_structure' parameter
        specification dict with defaults. For non-required arguments it checks
        if the default value provided conforms to the specified type, and if
        not - marks the argument as default, emitting a warning.

        Parameters
        ----------
        argument_name: str
            Name of the argument being processed.
        argument_spec: Dict[str, Any]
            Value of from the arguments_structure dict.

        Returns
        -------
        argument_spec: Dict[str, Any]
            Validated and filled-out specification.
        """
        default_spec = {
            "argparse_name": from_name_to_flag(argument_name),
            "description": "Description not provided.",
            "type": str,
            "nullable": True,
            "default": None,
            "required": False,
            "override": True,
            "enum": None,
        }
        spec = default_spec | argument_spec
        if not spec["required"] and not cls.verify_type(spec["default"], spec):
            spec["required"] = True
            KLogger.warning(
                f"Block class {cls.__name__} parameter {argument_name} was"
                " marked as not required, but the default value provided"
                f" ({spec['default']}) does not match the provided type"
                f" requirements (type={spec['type']},"
                f" nullable={spec['nullable']}, enum={spec['enum']})."
                " Parameter will be marked as required."
            )
        return spec

    @classmethod
    def build_from_config(
        cls, config: KenningBlockConfigDict, **kwargs: Dict[str, Any]
    ) -> Any:
        """
        Constructor wrapper that takes the parameters from config dict.

        This function matches parameters from the config, by name and type, to
        'arguments_structure' and then calls the constructor with all
        parameters that match (after necessary type conversions).

        Parameters
        ----------
        config: KenningBlockConfigDict
            Configuration dictionary.
        **kwargs : Dict[str, Any]
            Additional class-dependent arguments.

        Returns
        -------
        Any
            Instance created from provided config.

        Raises
        ------
        ConfigurationError
            Thrown, when there's a missing required parameter.
        """
        KLogger.debug(
            f"Building block class {cls.__name__} from standard Kenning block"
            " configuration dictionary..."
        )
        arguments_structure = cls.get_full_arguments_structure()
        final_args = {}
        class_specific_config = None
        direct_arguments = {}
        for block_type in config[BLOCK_CONFIGURATIONS_KEY].values():
            if cls.__name__ in block_type:
                class_specific_config = block_type[cls.__name__][
                    BLOCK_CONFIG_PARAMETERS_KEY
                ]
                direct_arguments = block_type[cls.__name__][
                    BLOCK_DIRECT_ARGUMENTS_KEY
                ]
                KLogger.debug(
                    f"Found block-specific config for {cls.__name__}"
                )
                break
        shared_config = config[UNAFFILIATED_PARAMETERS_KEY]
        missing_args = []
        for argument_name, spec in arguments_structure.items():
            spec = cls.process_argument_spec(argument_name, spec)
            alternative_argument_name = from_flag_to_name(
                spec["argparse_name"]
            )
            KLogger.debug(f"Matching argument {argument_name}: {spec}...")
            found = False
            value = None
            # 'shared_config' contains unmatched flags from argparse, while
            # 'class_specific_config' are flags, for that we already know which
            # block they belong to (and those in practice come from yaml/json
            # configs). Class arguments in Kenning have different names in
            # argparse and in yaml/json. Argparse names are stored under
            # 'argparse_name' key in arguments_structure in the form
            # '--arg-name', but after actually parsing the flags, they show up
            # in argparse.Namespace in the form 'arg_name'. Therefore we use
            # 'from_flag_to_name' above to convert and match
            # those properly.
            # In the code below, we first match from `class_specific_config`
            # and there we give matching priority to the regular argument name
            # (the one used in yaml/json - 'argument_name').
            # But then we match from `shared_config`, and there we match first
            # by the converted argparse name ('alternative_argument_name'),
            # because shared config flags typically come from argparse.
            # Matching priority is important, and changing it will break stuff,
            # since some arguments have argparse names that are the same as
            # other arguments' normal names.
            KLogger.debug("Matching by name...")
            if class_specific_config:
                if argument_name in class_specific_config.keys():
                    value = class_specific_config[argument_name]
                    found = True
                    KLogger.debug(f"Found {argument_name} in block config.")
                elif alternative_argument_name in class_specific_config.keys():
                    value = class_specific_config[alternative_argument_name]
                    found = True
                    KLogger.debug(
                        f"Found {alternative_argument_name} in block config."
                    )
            if not found or spec["override"]:
                if alternative_argument_name in shared_config.keys():
                    value = shared_config[alternative_argument_name]
                    found = True
                    KLogger.debug(
                        f"Found {alternative_argument_name} in unaffiliated"
                        " flags."
                    )
                elif argument_name in shared_config.keys():
                    value = shared_config[argument_name]
                    found = True
                    KLogger.debug(
                        f"Found {argument_name} in unaffiliated flags."
                    )
            if found:
                KLogger.debug(
                    f"Value for {argument_name} matched by name: {value}."
                )
                KLogger.debug("Matching by data type....")
            if found and cls.verify_type(value, spec):
                final_args[argument_name] = cls.convert_value(value, spec)
                KLogger.debug(
                    f"Type verified, {argument_name} value set: {value}"
                    f" ({final_args[argument_name]} after type conversion)."
                )
                continue
            elif not spec["required"]:
                final_args[argument_name] = cls.convert_value(
                    spec["default"], spec
                )
                KLogger.debug(
                    f"Matching by name and type failed for {argument_name},"
                    f" applying default value: {spec['default']}."
                )
                continue
            KLogger.error(
                f"Required {cls.__name__} parameter {argument_name} missing."
            )
            missing_args.append(argument_name)
        if len(missing_args) > 0:
            raise ConfigurationError(
                f"Block {cls.__name__} could not be constructed, due to"
                f" missing required parameters: {missing_args}."
            )
        cls_args = dict(final_args, **kwargs, **direct_arguments)
        KLogger.debug("Argument matching finished, building the block...")
        KLogger.debug(f"Direct arguments applied: {direct_arguments}")
        return cls(**cls_args)

    def to_json(self) -> Dict[str, Any]:
        """
        Convert object to JSON that contains its type and all parameters.

        Returns
        -------
        Dict[str, Any]
            JSON config of given object.
        """
        cls = self.__class__

        cls_parameters = {}
        for curr_cls in traverse_parents_with_args(cls):
            for arg_name, arg_opts in curr_cls.arguments_structure.items():
                if not hasattr(self, arg_name):
                    continue
                if "argparse_name" in arg_opts:
                    name = from_flag_to_name(arg_opts["argparse_name"])
                else:
                    name = arg_name

                cls_parameters[name] = getattr(self, arg_name)

        return {
            "type": f"{cls.__module__}.{cls.__name__}",
            "parameters": convert_to_jsontype(cls_parameters),
        }
