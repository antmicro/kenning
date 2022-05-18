"""
Module for preparing and serializing class arguments.
"""

import json
import jsonschema
import argparse
from typing import Dict
from pathlib import Path


"""
arguments_structure is a mapping (argument_name -> keywords)

Supported keywords:
argparse_name: Name that is prompted as an argparse argument. It is also
    used for JSON arguments with hyphens stripped from the prefix and
    changed to underscores.
    E.g. --some-name becomes some_name
description: Description of the argument.
type: Same as 'type' in argparse. The argument is converted to this value.
    Possible values for type: [int, float, str, bool, Path].

    Note that for bool types, the argument has to declare its default value.
default: Default value of the argument.
required: Determines whether argument is required.
enum: List of possible values of the argument.
is_list: Determines whether argument is a list of arguments.
    Possible values for is_list: [True, False].
    By default it is False.
    List of bool arguments is not supported

Example of an argument:
'compiled_model_path': {
    'argparse_name': '--compiled-model-path',
    'description': 'The path to the compiled model output',
    'type': Path,
    'required': True,
    'enum': ['/home/Documents', '/tmp']
}
"""


def convert_argparse_name(s):
    return s.lstrip('-').replace('-', '_')


def serialize(obj: object) -> str:
    """
    Serializes the given object into a JSON format.

    It serializes all variables mentioned in the dictionary
    returned by a `form_parameterschema` function.

    Note that object has to implement `form_parameterschema`
    which uses `add_parameterschema_argument` function
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
        properties = obj.form_parameterschema()['properties']
        to_serialize = {
            keywords['real_name']: name
            for name, keywords in properties.items()
        }
    else:
        return '{}'

    serialized_dict = {}

    for name, value in vars(obj).items():
        if name in to_serialize:
            serialized_dict[to_serialize[name]] = value

    class ArgumentEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Path):
                return str(obj)
            return json.JSONEncoder.default(self, obj)

    return json.dumps(serialized_dict, cls=ArgumentEncoder)


def get_parsed_json_dict(schema, json_dict):
    """
    Validates the given dictionary with the schema.
    Then it adds default values for missing
    arguments and converts the arguments to the appropriate types
    as jsonschema can not do that. Finally it provides new names
    that match the constructor arguments,

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

    # Including default values
    for name, keywords in schema['properties'].items():
        if name not in json_dict and 'default' in keywords:
            converted_json_dict[name] = keywords['default']
        elif name in json_dict:
            converted_json_dict[name] = json_dict[name]

    # Converting arguments to the final types
    for name, value in converted_json_dict.items():
        keywords = schema['properties'][name]

        if 'convert-type' in keywords:
            converter = keywords['convert-type']

            if 'type' in keywords and keywords['type'] == 'array':
                converted_json_dict[name] = [converter(v) for v in value]
            else:
                converted_json_dict[name] = converter(value)
        else:
            converted_json_dict[name] = value

    # Changing names so they match the constructor arguments
    converted_json_dict = {
        schema['properties'][name]['real_name']: value
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
        keywords = {}

        if 'type' in prop:
            if prop['type'] is bool:
                assert 'default' in prop and prop['default'] in [True, False]

                keywords['action'] = (
                    'store_false' if prop['default']
                    else 'store_true'
                )
            else:
                keywords['type'] = prop['type']
        if 'description' in prop:
            keywords['help'] = prop['description']
        if 'default' in prop:
            keywords['default'] = prop['default']
        if 'required' in prop and prop['required']:
            keywords['required'] = prop['required']
        if 'enum' in prop:
            keywords['choices'] = prop['enum']
        if 'is_list' in prop and prop['is_list']:
            keywords['nargs'] = '+'

        group.add_argument(argparse_name, **keywords)


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

        argparse_name = prop['argparse_name']
        argschema_name = convert_argparse_name(argparse_name)

        schema['properties'][argschema_name] = {}
        keywords = schema['properties'][argschema_name]

        keywords['real_name'] = name

        type_to_jsontype = {
            Path: 'string',
            str: 'string',
            float: 'number',
            int: 'integer',
            bool: 'boolean'
        }

        # Case for a list of keywords
        if 'is_list' in prop and prop['is_list']:
            keywords['type'] = 'array'

            if 'type' in prop:
                assert prop['type'] is not bool

                keywords['convert-type'] = prop['type']
                keywords['items'] = {
                    'type': type_to_jsontype[prop['type']]
                }
        # Case for a single argument
        else:
            if 'type' in prop:
                keywords['convert-type'] = prop['type']
                keywords['type'] = type_to_jsontype[prop['type']]

        if 'description' in prop:
            keywords['description'] = prop['description']
        if 'default' in prop:
            keywords['default'] = prop['default']
        if 'required' in prop and prop['required']:
            if 'required' not in schema:
                schema['required'] = []
            schema['required'].append(argschema_name)
        if 'enum' in prop:
            keywords['enum'] = prop['enum']
