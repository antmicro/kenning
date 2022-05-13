"""
Module for preparing and serializing class arguments.
"""

import json
import jsonschema
import argparse
from typing import Dict
from pathlib import Path


def serialize(obj: object) -> str:
    """
    Serializes the given object into a JSON format.

    It serializes all variables mentioned in the dictionary
    returned by a `form_parameterschema` function.

    Note that object has to implement `form_parameterschema`
    to be serializable.

    Parameters
    ----------
    obj : object
        Object to serialize

    Returns
    -------
    str: Serialized object in a valid JSON format.
    """
    if hasattr(obj, 'form_parameterschema'):
        to_serialize = obj.form_parameterschema()['properties'].keys()
    else:
        return '{}'

    serialized_dict = {}

    for name, value in vars(obj).items():
        if name in to_serialize:
            serialized_dict[name] = value

    class ArgumentEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Path):
                return str(obj)
            return json.JSONEncoder.default(self, obj)

    return json.dumps(serialized_dict, cls=ArgumentEncoder)


"""
arguments_structure is a mapping (argument_name -> keywords)

Supported keywords:
argparse_name: Name that is prompted as an argparse argument.
description: Description of the argument.
type: Same as 'type' in argparse. The argument is converted to this value.
    Possible values for type: [int, float, str, bool, Path]
    Note that for bool types, the argument has to declare its default value.
default: Default value of the argument.
required: Defines whether argument is required.
enum: List of possible values of the argument.

Example of an argument:
'compiled_model_path': {
    'argparse_name': '--compiled-model-path',
    'description': 'The path to the compiled model output',
    'type': Path,
    'required': True,
    'enum': ['/home/Documents', '/tmp']
}
"""


def get_parsed_json_dict(schema, json_dict):
    """
    Validates the given dictionary with the schema.
    Then it adds default values for missing
    arguments and converts the arguments to the
    appropriate types, as jsonschema can not do that.

    Parameters
    ----------
    schema : Dict
        Schema to validate with
    json_dict : Dict
        Dictionary to validate
    """
    jsonschema.validate(
        instance=json_dict,
        schema=schema
    )

    converted_json_dict = {}
    for name, keywords in schema['properties'].items():
        if name not in json_dict and 'default' in keywords:
            converted_json_dict[name] = keywords['default']
        elif name in json_dict:
            converted_json_dict[name] = json_dict[name]

    converted_json_dict = {
        name: (
            schema['properties'][name]['convert-type'](str(value))
            if 'convert-type' in schema['properties'][name]
            else value
        )
        for name, value in converted_json_dict.items()
    }

    return converted_json_dict


def add_argparse_argument(
        group: argparse._ArgumentGroup,
        struct: Dict[str, Dict],
        *names: str):
    """
    Adds arguments to the argparse group based on the given
    struct and names. If no names are given it uses all
    properties of the struct.

    Note that the function modifies the given group.

    Parameters
    ----------
    group : argparse._ArgumentGroup
        Argparse group that is filled with new properties.
    struct : Dict[str, Dict]
        Struct with properties described.
    *names : str
        Names of the properties that are to be added to the group.
        If empty every property in struct is used.
    """
    if not names:
        names = struct.keys()

    for name in names:
        prop = struct[name]

        argparse_name = prop['argparse_name']
        arguments = {}

        if 'type' in prop:
            if prop['type'] is bool:
                arguments['action'] = (
                    'store_true' if not prop['default']
                    else 'store_false'
                )
            else:
                arguments['type'] = prop['type']
        if 'description' in prop:
            arguments['help'] = prop['description']
        if 'default' in prop:
            arguments['default'] = prop['default']
        if 'required' in prop and prop['required']:
            arguments['required'] = prop['required']
        if 'enum' in prop:
            arguments['choices'] = prop['enum']

        group.add_argument(argparse_name, **arguments)


def add_parameterschema_argument(
        schema: Dict,
        struct: Dict[str, Dict],
        *names: str):
    """
    Adds arguments to the schema based on the given
    struct and names. If no names are given it uses all
    properties of the struct.

    Note that the function modifies the given schema.

    Parameters
    ----------
    schema : Dict
        Schema that is filled with new properties.
    struct : Dict[str, Dict]
        Struct with properties described.
    *names : str
        Names of the properties that are to be added to the group.
        If empty every property in struct is used.
    """
    if 'properties' not in schema:
        schema['properties'] = {}

    if not names:
        names = struct.keys()

    for name in names:
        prop = struct[name]

        schema['properties'][name] = {}
        arguments = schema['properties'][name]

        type_to_jsontype = {
            Path: 'string',
            str: 'string',
            float: 'number',
            int: 'number',
            bool: 'boolean'
        }

        if 'type' in prop:
            arguments['convert-type'] = prop['type']
            arguments['type'] = type_to_jsontype[prop['type']]
        if 'description' in prop:
            arguments['description'] = prop['description']
        if 'default' in prop:
            arguments['default'] = prop['default']
        if 'required' in prop and prop['required']:
            if 'required' not in schema:
                schema['required'] = []
            schema['required'].append(name)
        if 'enum' in prop:
            arguments['enum'] = prop['enum']
