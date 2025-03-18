# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for preparing and serializing class arguments.
"""

import argparse
import json
import os.path
from abc import ABC
from functools import partial
from pathlib import Path
from types import UnionType
from typing import (
    Any,
    Callable,
    Dict,
    GenericAlias,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import jsonschema
import numpy as np

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
    "subcommands",
    # AutoML specific keys
    "AutoML",
    "list_range",
    "item_range",
]


def from_argparse_name(s):
    """
    Converts argparse name to snake-case name.
    """
    return s.lstrip("-").replace("-", "_")


def to_argparse_name(s: str) -> str:
    """
    Converts entry from arguments_structure to argparse entry.
    """
    return "--" + s.replace("_", "-")


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
            raise argparse.ArgumentError(
                None, f"missing required arguments: {', '.join(missing_args)}"
            )

    if args.json_cfg is not None and any(flag_config_not_none):
        raise argparse.ArgumentError(
            None,
            "JSON and flag configurations are mutually exclusive. "
            "Please use only one method of configuration.",
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


class ArgsManagerConvertError(Exception):
    """
    Raised when provided value cannot be converted
    by any of specified converters.
    """

    ...


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
        if name not in json_dict and "default" in keywords:
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

        if "type" in keywords and "array" in keywords["type"]:
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


def get_parsed_args_dict(cls: type, args: argparse.Namespace) -> Dict:
    """
    Converts namespace provided by arguments parser into dictionary.

    Parameters
    ----------
    cls : type
        Class of object being parsed.
    args : argparse.Namespace
        Namespace provided by arguments parser.

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
    classes = [cls]
    args_structure = {}
    while len(classes):
        curr_cls = classes.pop(0)
        classes.extend(curr_cls.__bases__)
        if not hasattr(curr_cls, "arguments_structure"):
            continue
        args_structure = dict(args_structure, **curr_cls.arguments_structure)

    # parse arguments
    parsed_args = {}
    for arg_name, arg_properties in args_structure.items():
        if "argparse_name" in arg_properties:
            argparse_name = from_argparse_name(arg_properties["argparse_name"])
        else:
            argparse_name = arg_name

        if hasattr(args, argparse_name):
            value = getattr(args, argparse_name)
        else:
            try:
                value = arg_properties["default"]
            except KeyError:
                raise Exception(
                    f"No default value provided for {argparse_name}"
                )

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
        if "type" in arg_properties and value is not None:
            value = arg_properties["type"](value)

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
    if not isinstance(_type, GenericAlias):
        return _type, None
    main_type = _type.__origin__
    sub_types = [
        arg.__args__ if isinstance(arg, UnionType) else arg
        for arg in _type.__args__
    ]
    return main_type, sub_types


def add_argparse_argument(
    group: argparse._ArgumentGroup,
    struct: Dict[str, Dict],
    args: argparse.Namespace,
    *names: str,
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

    Raises
    ------
    KeyError :
        Raised if there is a keyword that is not recognized.
    """
    from kenning.cli.config import AVAILABLE_COMMANDS, get_used_subcommands

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
            argparse_name = to_argparse_name(name)

        keywords = {}
        if "type" in prop:
            prop_type, prop_sub_types = get_type(prop["type"])
            if prop_type is bool:
                assert "default" in prop and prop["default"] in [True, False]

                keywords["action"] = (
                    "store_false" if prop["default"] else "store_true"
                )
            elif prop_type is list:
                keywords["nargs"] = "+"
                converters = set()
                for sub_type in prop_sub_types:
                    if isinstance(sub_type, (list, tuple)):
                        converters.update(sub_type)
                    else:
                        converters.add(sub_type)
                keywords["type"] = partial(convert, converters=converters)
            else:
                keywords["type"] = prop_type
        if "description" in prop:
            keywords["help"] = prop["description"]
        if "default" in prop:
            keywords["default"] = prop["default"]
        if "required" in prop and prop["required"]:
            keywords["required"] = prop["required"]
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
            argschema_name = from_argparse_name(argparse_name)
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
            if prop_type is list and prop_sub_type:
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
        if "enum" in prop:
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
        classes = [cls]
        parameterschema = {"type": "object", "additionalProperties": False}

        while len(classes):
            curr_cls = classes.pop(0)
            classes.extend(curr_cls.__bases__)
            if not hasattr(curr_cls, "arguments_structure"):
                continue
            add_parameterschema_argument(
                parameterschema, curr_cls.arguments_structure
            )

        return parameterschema

    @classmethod
    def form_argparse(
        cls, args: argparse.Namespace
    ) -> Tuple[argparse.ArgumentParser, Optional[argparse._ArgumentGroup]]:
        """
        Creates argparse parser based on `arguments_structure` of class and its
        all parent classes.

        Parameters
        ----------
        args : argparse.Namespace
            Arguments from ArgumentParser object.

        Returns
        -------
        Tuple[argparse.ArgumentParser, Optional[argparse._ArgumentGroup]]
            Tuple with the argument parser object that can act as parent for
            program's argument parser, and the corresponding arguments' group
            pointer.
        """
        classes = [cls]
        parser = argparse.ArgumentParser(
            add_help=False, conflict_handler="resolve"
        )
        group = None

        while len(classes):
            curr_cls = classes.pop(0)
            classes.extend(curr_cls.__bases__)
            if not hasattr(curr_cls, "arguments_structure"):
                continue
            group = parser.add_argument_group(
                title=f"{curr_cls.__name__} arguments"
            )
            add_argparse_argument(group, curr_cls.arguments_structure, args)

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

    def to_json(self) -> Dict[str, Any]:
        """
        Convert object to JSON that contains its type and all parameters.

        Returns
        -------
        Dict[str, Any]
            JSON config of given object.
        """
        cls = self.__class__

        classes = [cls]
        cls_parameters = {}
        while len(classes):
            curr_cls = classes.pop(0)
            classes.extend(curr_cls.__bases__)
            if not hasattr(curr_cls, "arguments_structure"):
                continue
            for arg_name, arg_opts in curr_cls.arguments_structure.items():
                if not hasattr(self, arg_name):
                    continue
                if "argparse_name" in arg_opts:
                    name = from_argparse_name(arg_opts["argparse_name"])
                else:
                    name = arg_name

                cls_parameters[name] = getattr(self, arg_name)

        return {
            "type": f"{cls.__module__}.{cls.__name__}",
            "parameters": convert_to_jsontype(cls_parameters),
        }
