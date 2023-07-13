#!/usr/bin/env python

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
A script that provides information about a given kenning class.

More precisely, it displays:
* module and class docstring
* imported dependencies, including information if they are available or not
* supported input and output formats (lots of the classes provide such
  information one way or the other)
* node's parameters, with their help and default values
"""
import argparse
import ast
import importlib
import inspect
import os.path
import sys
from typing import Union, List, Tuple, Optional, Dict
from kenning.cli.command_template import (
    CommandTemplate, GROUP_SCHEMA, INFO)
from kenning.utils import logger
from pathlib import Path
from typing import Union, List, Dict

import astunparse
from isort import place_module

from kenning.core.model import ModelWrapper
from kenning.utils.args_manager import to_argparse_name

KEYWORDS = ['inputtypes', 'outputtypes', 'arguments_structure']


class Argument:
    """
    Class representing an argument. Fields that are empty are not displayed.
    """

    def __init__(self):
        self.name = ''
        self.argparse_name = ''
        self.description = ''
        self.required = ''
        self.default = ''
        self.nullable = ''
        self.type = ''
        self.enum: List[str] = []

    def __repr__(self):
        lines = [f'* {self.name}']

        if self.argparse_name:
            lines.append(f'  * argparse name: {self.argparse_name}')
        if self.type:
            lines.append(f'  * type: {self.type}')
        if self.description:
            lines.append(f'  * description: {self.description}')
        if self.required:
            lines.append(f'  * required: {self.required}')
        if self.default:
            lines.append(f'  * default: {self.default}')
        if self.nullable:
            lines.append(f'  * nullable: {self.nullable}')

        if len(self.enum) != 0:
            lines.append('  * enum')
        for element in self.enum:
            lines.append(f'    * {element}')

        return '\n'.join(lines)


def get_class_module_name(syntax_node: Union[ast.ClassDef, ast.
                          Module]) -> str:
    """
    Displays class name from syntax node

    Parameters
    ----------
    syntax_node: Union[ast.ClassDef, ast.Module]
        Class syntax node

    Returns
    -------
    str: Formatted markdown-like string to be printed later.
    """
    if isinstance(syntax_node, ast.ClassDef):
        return f'Class: {syntax_node.name}\n\n'


def get_class_module_docstrings(syntax_node: Union[ast.ClassDef, ast. Module]) -> str:  # noqa: E501
    """
    Displays docstrings of provided class or module

    Parameters
    ----------
    syntax_node: Union[ast.ClassDef, ast.Module]
        Syntax node representing a class or module

    Returns
    -------
    str: Formatted markdown-like string to be printed later.
    """

    docstring = ast.get_docstring(syntax_node, clean=True)

    if not docstring:
        return f'Class: {syntax_node.name}\n\n'

    docstring = '\n'.join(
        ['    ' + docstr for docstr in docstring.strip('\n').split('\n')])

    if isinstance(syntax_node, ast.ClassDef):
        return f'Class: {syntax_node.name}\n\n{docstring}\n\n'

    if isinstance(syntax_node, ast.Module):
        return f'Module description:\n\n{docstring}\n\n'


def get_dependency(syntax_node: Union[ast.Import, ast.ImportFrom]) \
        -> str:
    """
    Extracts a dependency from an import syntax node and checks whether the
    dependency is satisfied. It also skips internal kenning modules

    Parameters
    ----------
    syntax_node: Union[ast.Import, ast.ImportFrom]
        An assignment like `from iree.compiler import version``

    Returns
    -------
    str: Formatted markdown-like string to be printed later. Empty strings
    represent dependencies that were skipped - either they belong to kenning
    or are provided by the default python distribution
    """
    for dependency in syntax_node.names:
        module_path = ''
        dependency_path = ''
        if isinstance(syntax_node, ast.ImportFrom):
            dependency_path = f'{syntax_node.module}.{dependency.name}'
            module_path = f'{syntax_node.module}'

        if isinstance(syntax_node, ast.Import):
            dependency_path = f'{dependency.name}'
            module_path = dependency_path

        if module_path == '' or dependency_path == '':
            return ''

        try:
            importlib.import_module(module_path)

            if 'kenning' in dependency_path:
                return ''

            if place_module(module_path) == 'STDLIB':
                return ''

            return '* ' + dependency_path + '\n'
        except (ImportError, ModuleNotFoundError, Exception) as e:
            return f'* {dependency_path} - Not available (Reason: {e})\n'


def get_input_specification(syntax_node: ast.Assign) -> str:
    """
    Displays information about the input specification as bullet points

    Parameters
    ----------
    syntax_node: ast.Assign
        An assignment like `inputtypes = []`

    Returns
    -------
    str: Formatted markdown-like string to be printed later.
    """

    input_formats = ''

    if isinstance(syntax_node.value, ast.List) \
            and len(syntax_node.value.elts) == 0:
        return ''

    if isinstance(syntax_node.value, ast.List):
        for input_format in syntax_node.value.elts:
            input_formats += f'* {input_format.value}\n'
        return input_formats

    for input_format in syntax_node.value.keys:
        input_formats += f'* {input_format.value}\n'

    return input_formats


def parse_dict_node_to_string(dict_node: ast.Dict) -> List[str]:
    """
    Parses an ast.Dict to a nicely formatted list of strings in markdown
    format.

    Parameters
    ----------
    dict_node: ast.Dict
        AST dict node to extract the data from

    Returns
    -------
    List[str]: List of formatted markdown-like strings to be printed later.
    """

    # formatted lines to be returned
    resulting_output = []

    dict_elements = []
    for key, value in zip(dict_node.keys, dict_node.values):

        if not isinstance(value, ast.List):
            resulting_output.append(f'* {key.value}: {value.value}')
            continue

        [dict_elements.append(element) for element in value.elts]

    for dict_element in dict_elements:

        resulting_output.append(f'* {dict_element.values[0].value}\n')

        for key, value in zip(dict_element.keys[1:], dict_element.values[1:]):
            resulting_output.append(f'  * {key.value}: '
                                    f'{clean_variable_name(value)}\n')

    return resulting_output


def get_io_specification(class_node: ast.ClassDef) -> List[str]:
    """
    Extracts io_specification when classes specify io this way.

    Parameters
    ----------
    class_node: ast.ClassDef
        AST class node to find and extract io_specification from

    Returns
    -------
    List[str]: List of formatted markdown-like strings to be printed later.
    """
    io_spec_function_node = None

    if len(class_node.body) <= 0:
        return []

    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef):
            continue

        if node.name != '_get_io_specification':
            continue

        io_spec_function_node = node

    if io_spec_function_node is None or len(io_spec_function_node.body) <= 0:
        return []

    io_spec_dict_node = None
    for node in io_spec_function_node.body:
        if not isinstance(node, ast.Return):
            continue

        io_spec_dict_node = node.value

    if io_spec_dict_node is None:
        return []

    return parse_dict_node_to_string(io_spec_dict_node)


def get_output_specification(syntax_node: ast.Assign) -> str:
    """
    Displays information about the output specification as bullet points

    Parameters
    ----------
    syntax_node: ast.Assign
        An assignment like `outputtypes = ['iree']`

    Returns
    -------
    str: Formatted markdown-like string to be printed later.
    """
    for output_format in syntax_node.value.elts:
        return f'* {output_format.value}\n'


def clean_variable_name(variable_name: ast.AST) -> str:
    """
    Unparses and cleans a parsed variable name as string from single quotation
    marks and trailing whitespaces

    Parameters
    ----------
    variable_name: ast.AST
        Variable to be cleaned up, e.g. "'tflite' "

    Returns
    -------
    str: Cleaned up variable
    """
    return astunparse \
        .unparse(variable_name) \
        .strip() \
        .removeprefix("'") \
        .removesuffix("'")


def get_arguments_structure(syntax_node: ast.Assign, source_path: str) \
        -> str:
    """
    Displays information about the argument structure specification as
    bullet points

    Parameters
    ----------
    syntax_node: ast.Assign
        An assignment like `arguments_structure = {'compiler_args': {}}`
    source_path: str
        Source path of the code to be parsed

    Returns
    -------
    str: Formatted markdown-like string to be printed later.
    """
    output_string = ''

    for argument, argument_specification_dict in zip(syntax_node.value.keys,
                                                     syntax_node.value.values):
        argument_object = Argument()

        argument_object.name = argument.value

        for key, value in zip(argument_specification_dict.keys,
                              argument_specification_dict.values):

            if isinstance(value, ast.Call) \
                    and isinstance(value.func, ast.Name) \
                    and value.func.id == 'list':
                argument_list_variable = astunparse.unparse(value) \
                    .strip() \
                    .removeprefix("'") \
                    .removesuffix("'") \
                    .replace('list(', '') \
                    .replace('.keys())', '')

                argument_keys, argument_type = evaluate_argument_list_of_keys(
                    argument_list_variable,
                    source_path)

                argument_object.enum = argument_keys
                argument_object.type = argument_type
            elif isinstance(value, ast.Call) \
                    and isinstance(value.func, ast.Attribute):
                key_str = clean_variable_name(key)
                value_str = clean_variable_name(value)

                argument_object.__setattr__(key_str, [value_str])

            elif key.value == 'enum':
                argument_list_variable = clean_variable_name(value)

                enum_list, argument_type = evaluate_argument_list(
                    argument_list_variable,
                    source_path)

                argument_object.enum = enum_list
                argument_object.type = argument_type

            else:
                key_str = clean_variable_name(key)
                value_str = clean_variable_name(value)

                argument_object.__setattr__(key_str, value_str)

        output_string += argument_object.__repr__() + '\n'

    return output_string


def evaluate_argument_list_of_keys(argument_list_name: str, source_path: str) \
        -> Tuple[List[str], str]:
    """
    Evaluate an expression like `list(some_dict.keys())` and return the list
    of elements as strings.

    Parameters
    ----------
    argument_list_name: str
        Variable name that the list of keys is assigned to
    source_path:
        Path of the code to be parsed

    Returns
    -------
    Tuple[List[str], str]: tuple with the first argument being the list of
    evaluated elements and the second being the type as a string
    """
    with open(source_path, 'r') as file:
        parsed_file = ast.parse(file.read())

    syntax_nodes = ast.walk(parsed_file)

    argument_list_keys = []
    argument_type = ''

    for node in syntax_nodes:
        if not isinstance(node, ast.Assign):
            continue

        if not isinstance(node.targets[0], ast.Name):
            continue

        if not node.targets[0].id == argument_list_name:
            continue

        for key in node.value.keys:
            argument_list_keys.append(key.value)

        argument_type = f'List[{type(node.value.keys[0].value).__name__}]'

        break

    return argument_list_keys, argument_type


def evaluate_argument_list(argument_list_name: str, source_path: str) \
        -> Tuple[List[str], str]:
    """
    Evaluate an expression like `list('tflite', 'tvm')` and return the list
    of elements as strings.

    Parameters
    ----------
    argument_list_name: str
        Variable name that the list of elements is assigned to.
    source_path:
        Path of the code to be parsed

    Returns
    -------
    Tuple[List[str], str]: tuple with the first argument being the list of
    evaluated elements and the second being the type as a string
    """
    with open(source_path) as file:
        parsed_file = ast.parse(file.read())

    syntax_nodes = ast.walk(parsed_file)

    enum_elements = []
    argument_type = ''

    # argument list is an explicit python list (['int8', 'float16'])
    if argument_list_name.endswith(']') and argument_list_name[0] == '[':
        enum_elements = eval(argument_list_name)
        if len(enum_elements) > 0:
            argument_type = f'List[{type(enum_elements[0]).__name__}]'
        else:
            argument_type = 'List[]'
        return enum_elements, argument_type

    for node in syntax_nodes:
        if not isinstance(node, ast.Assign):
            continue

        if not isinstance(node.targets[0], ast.Name):
            continue

        if not node.targets[0].id == argument_list_name:
            continue

        for element in node.value.elts:
            enum_elements.append(element.value)

        argument_type = f'List[{type(node.value.elts[0].value).__name__}]'
        break

    return enum_elements, argument_type


def get_args_structure_from_parameterschema(parameterschema) -> List[str]:
    resulting_lines = []

    args_structure = parameterschema['properties']

    if not args_structure:
        return ['']

    for arg_name, arg_dict in args_structure.items():
        resulting_lines.append(f'* {arg_name}\n')

        resulting_lines.append(f'  * argparse_name: '
                               f'{to_argparse_name(arg_name)}\n')
        for key, value in arg_dict.items():
            # skip real_name as it is the same as arg_name
            if key == 'real_name':
                continue

            # expand enums (lists)
            if isinstance(value, list):
                resulting_lines.append(f'  * {key}\n')
                for elt in value:
                    resulting_lines.append(f'    * {elt}\n')
                continue

            # extract qualified class name if value is a class
            if inspect.isclass(value):
                resulting_lines.append(f'  * {key}: {value.__module__}.'
                                       f'{value.__qualname__}\n')
                continue

            resulting_lines.append(f'  * {key}: {value}\n')

    return resulting_lines


def parse_io_spec_dict_to_str(dictionary: Dict) -> List[str]:  # noqa E501
    """
    Recursively parses a dictionary to a list of formatted, markdown-like strings  # noqa E501

    Parameters
    ----------
    dictionary: Dict
        A python dictionary to be parsed

    Return
    ------
    List[str]: A list of formatted, markdown-like strings
    """
    resulting_output = []

    dict_elements = []

    for key, value in dictionary.items():
        if not isinstance(value, list):
            resulting_output.append(f'* {key}: {value}\n')

        [dict_elements.append(elt) for elt in value]

    for dict_element in dict_elements:
        resulting_output.append(f'* {dict_element["name"]}\n')
        dict_element.pop('name', None)

        for key, value in dict_element.items():
            if isinstance(value, list):
                resulting_output.append(f'  * {key}\n')
                [resulting_output.append(f'    * {elt}\n') for elt in value]
                continue

            resulting_output.append(f'  * {key}: {value}\n')

    return resulting_output


def generate_class_info(target: str, class_name='', docstrings=True,
                        dependencies=True, input_formats=True,
                        output_formats=True, argument_formats=True,
                        load_class_with_args=[]) \
        -> List[str]:
    """
    Wrapper function that handles displaying information about a class

    Parameters
    ----------
    target: str
        Target class path or module name e.g. either `kenning.core.flow` or
         `kenning/core/flow.py`
    class_name: str
        Name of a specific class to display information about
    docstrings: bool
        Flag whether to display docstrings or not
    dependencies: bool
        Flag whether to display dependencies and their availability
    input_formats: bool
        Flag whether to display input formats
    output_formats: bool
        Flag whether to display output formats
    argument_formats: bool
        Flag whether to display argument formats
    load_class_with_args: List[str]
        # TODO

    Returns
    -------
    List[str]: List of formatted, markdown-like lines to be printed
    """
    resulting_lines = []

    if class_name is None:
        class_name = ''

    # if target contains a class, split to path and class name
    split_target = target.split('.')
    if split_target[-1][0].isupper():
        class_name = split_target[-1]
        split_target = split_target[:-1]

    target = '.'.join(split_target)

    target_path = target
    if not target.endswith('.py'):
        target_path = target.replace('.', '/')
        target_path += '.py'

    if not os.path.exists(target_path):
        return [f'File {target_path} does not exist\n']

    with open(target_path, 'r') as file:
        parsed_file = ast.parse(file.read())

    syntax_nodes = ast.walk(parsed_file)

    class_nodes = []
    dependency_nodes = []

    input_specification_node = None
    output_specification_node = None
    arguments_structure_node = None

    found_io_specification = False
    io_specification_lines = {}

    imported_class = None
    parameterschema = None
    class_object = None

    if class_name != '':
        # try to load the class into memory
        module_path = path = target_path[:-3].replace('/', '.')
        try:
            if class_name == '':
                return [
                    'Provide a class name in the module-like path when trying '
                    'to load a class with arguments']

            imported_class = getattr(
                importlib.import_module(path),
                class_name
            )

            parameterschema = imported_class.form_parameterschema()

        except (ModuleNotFoundError, ImportError, Exception) as e:
            return [f'Cannot import class {class_name} from {module_path}\n'
                    f'Reason: {e}']

    if imported_class:

        if issubclass(imported_class, ModelWrapper):
            # create a temporary directory for the dataset
            dataset_path = Path('build/tmp-dataset')
            model_path = Path(imported_class.pretrained_modelpath)

            dataset_path.touch(exist_ok=False)
            dataset = imported_class.default_dataset(
                Path(dataset_path), download_dataset=True)

            class_object = imported_class(model_path, dataset, from_file=True)

            dataset_path.unlink(missing_ok=True)

    for node in syntax_nodes:
        if isinstance(node, ast.ClassDef) and class_name != '' \
                and node.name == class_name:
            class_nodes.append(node)

            if not imported_class:
                io_specification = get_io_specification(node)
                if len(io_specification) > 0:
                    io_specification_lines[node] = io_specification
                    found_io_specification = True

        if isinstance(node, ast.ClassDef) and class_name == '':
            class_nodes.append(node)

            if not imported_class:
                io_specification = get_io_specification(node)
                if len(io_specification) > 0:
                    io_specification_lines[node] = io_specification
                    found_io_specification = True

        if isinstance(node, ast.Module) and class_name == '':
            class_nodes.append(node)

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            dependency_nodes.append(node)

        if isinstance(node, ast.Assign) and \
                isinstance(node.targets[0], ast.Name):
            if node.targets[0].id not in KEYWORDS:
                continue

            if node.targets[0].id == KEYWORDS[0]:
                input_specification_node = node
            if node.targets[0].id == KEYWORDS[1]:
                output_specification_node = node

            if node.targets[0].id == KEYWORDS[2]:
                arguments_structure_node = node

    if docstrings:
        if len(class_nodes) == 0:
            resulting_lines.append(f'Class {class_name} has not been found')
            return resulting_lines

        for node in class_nodes:
            resulting_lines.append(get_class_module_docstrings(node))
    else:
        for node in class_nodes:
            resulting_lines.append(get_class_module_name(node))

        if node in io_specification_lines.keys() \
                and (input_formats or output_formats):
            # i/o specification found, extract from static source code
            resulting_lines.append('Input/output specification:\n')
            resulting_lines += io_specification_lines[node]
            resulting_lines.append('\n')

        if imported_class and (input_formats or output_formats):
            # i/o specification found, extract by creating an object
            resulting_lines.append('Input/output specification:\n')

            io_spec = class_object.get_io_specification()
            resulting_lines += parse_io_spec_dict_to_str(io_spec)

    if dependencies:
        resulting_lines.append('Dependencies:\n')
        dependencies: List[str] = []
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        for node in dependency_nodes:
            dependency_str = get_dependency(node)
            if dependency_str == '':
                continue
            dependencies.append(dependency_str)

        for dep_str in list(set(dependencies)):
            resulting_lines.append(dep_str)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        resulting_lines.append('\n')

    if input_formats and not found_io_specification:
        resulting_lines.append('Input formats:\n')
        if input_specification_node:
            data = get_input_specification(input_specification_node)
            resulting_lines.append(data if data else 'No inputs')
        else:
            resulting_lines.append('No inputs')
        resulting_lines.append('\n\n')

    if output_formats and not found_io_specification:
        resulting_lines.append('Output formats:\n')
        if output_specification_node:
            data = get_output_specification(output_specification_node)
            resulting_lines.append(data if data else 'No outputs')
        else:
            resulting_lines.append('No outputs')
        resulting_lines.append('\n\n')

    if argument_formats:
        resulting_lines.append('Arguments specification:\n')
        if arguments_structure_node:
            data = get_arguments_structure(
                arguments_structure_node,
                target_path
            )
            resulting_lines.append(data if data else 'No arguments')
        else:
            resulting_lines.append('No arguments')
        resulting_lines.append('\n\n')

        if parameterschema:
            resulting_lines += \
                get_args_structure_from_parameterschema(parameterschema)

        elif arguments_structure_node:
            resulting_lines.append(get_arguments_structure(
                arguments_structure_node, target_path))
    return resulting_lines


class ClassInfoRunner(CommandTemplate):
    parse_all = True
    description = __doc__.split('\n\n')[0]

    @staticmethod
    def configure_parser(
        parser: Optional[argparse.ArgumentParser] = None,
        command: Optional[str] = None,
        types: List[str] = [],
        groups: Dict[str, argparse._ArgumentGroup] = None,
    ) -> Tuple[argparse.ArgumentParser, Dict]:
        parser, groups = super(
            ClassInfoRunner, ClassInfoRunner
        ).configure_parser(
            parser,
            command,
            types,
            groups
        )

        info_group = parser.add_argument_group(GROUP_SCHEMA.format(INFO))

        info_group.add_argument(
            'target',
            help='Module-like path of the module or class '
                 '(e.g. kenning.compilers.onnx)',
            type=str
        )
        info_group.add_argument(
            '--docstrings',
            help='Display class docstrings',
            action='store_true'
        )
        info_group.add_argument(
            '--dependencies',
            help='Display class dependencies',
            action='store_true'
        )
        info_group.add_argument(
            '--input-formats',
            help='Display class input formats',
            action='store_true'
        )
        info_group.add_argument(
            '--output-formats',
            help='Display output formats',
            action='store_true'
        )
        info_group.add_argument(
            '--argument-formats',
            help='Display the argument specification',
            action='store_true'
        )
        info_group.add_argument(
            '--load-class-with-args',
            help='',
            nargs='*'
        )
        return parser, groups

    @staticmethod
    def run(args: argparse.Namespace, **kwargs):
        logger.set_verbosity(args.verbosity)

        args = {k: v for k, v in vars(args).items() if v is not None}  # noqa: E501

        # if no flags are given, set all of them to True (display everything)
        if not any([v for v in args.values() if type(v) is bool]):
            for k, v in args.items():
                args[k] = True if type(v) is bool else v
        resulting_output = generate_class_info(**args)

        for result_line in resulting_output:
            print(result_line, end='')


def main(argv):
    parser, _ = ClassInfoRunner.configure_parser(command=argv[0])
    args, _ = parser.parse_known_args(argv[1:])

    ClassInfoRunner.run(args)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    ret = main(sys.argv)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    sys.exit(ret)

