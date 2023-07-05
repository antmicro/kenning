#!/usr/bin/env python

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
A script that provides information about a given kenning class.

More precisely, it displays:
    - module and class docstring
    - imported dependencies, including information if they are available or not
    - supported input and output formats (lots of the classes provide such
     information one way or the other)
    - node's parameters, with their help and default values
"""
import argparse
import ast
import importlib
import os.path
import sys
from typing import Union, List

import astunparse
from isort import place_module

from kenning.utils.class_loader import load_class

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


def print_class_module_docstrings(syntax_node: Union[ast.ClassDef, ast.
                                  Module]) -> str:
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
        return f'Class: {syntax_node.name}\n'

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
        except ImportError or ModuleNotFoundError as e:
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
    return astunparse\
        .unparse(variable_name) \
        .strip() \
        .removeprefix("'") \
        .removesuffix("'")


def print_arguments_structure(syntax_node: ast.Assign, source_path: str)\
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
        -> tuple[List[str], str]:
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
    tuple[List[str], str]: tuple with the first argument being the list of
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
        -> tuple[List[str], str]:
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
    tuple[List[str], str]: tuple with the first argument being the list of
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


def generate_class_info(target: str, class_name='', docstrings=True,
                        dependencies=True, input_formats=True,
                        output_formats=True, argument_formats=True)\
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

    for node in syntax_nodes:
        if isinstance(node, (ast.ClassDef, ast.Module)):
            if isinstance(node, ast.ClassDef) \
                    and class_name != '' \
                    and node.name == class_name:
                class_nodes.append(node)

            if isinstance(node, ast.ClassDef) and class_name == '':
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
            resulting_lines.append(print_class_module_docstrings(node))
    else:
        for node in class_nodes:
            resulting_lines.append(get_class_module_name(node))

    if dependencies:
        resulting_lines.append('Dependencies:\n')
        dependencies: List[str] = []
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        for node in dependency_nodes:
            dependency_str = get_dependency(node)
            if dependency_str == '':
                continue
            dependencies.append(dependency_str)

        [resulting_lines.append(dep_str)
         for dep_str in list(set(dependencies))]

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        resulting_lines.append('\n')

    if input_formats:
        resulting_lines.append('Input formats:\n')
        if input_specification_node:
            resulting_lines.append(get_input_specification(
                input_specification_node))
        # print('')
        resulting_lines.append('\n')

    if output_formats:
        resulting_lines.append('Output formats:\n')
        if output_specification_node:
            resulting_lines.append(get_output_specification(
                output_specification_node))
        resulting_lines.append('\n')

    if argument_formats:
        resulting_lines.append('Arguments specification:\n')
        if arguments_structure_node:
            resulting_lines.append(print_arguments_structure(
                arguments_structure_node, target_path))

    return resulting_lines


def main(argv):
    parser = argparse.ArgumentParser(argv[0],
                                     description='Provides information about a'
                                                 ' given kenning module or'
                                                 ' class. If no flags are'
                                                 ' given, displays'
                                                 ' the full output')

    parser.add_argument(
        'target',
        help='Module-like path of the module or class '
             '(e.g. kenning.compilers.onnx)',
        type=str
    )
    parser.add_argument(
        '--docstrings',
        help='Display class docstrings',
        action='store_true'
    )
    parser.add_argument(
        '--dependencies',
        help='Display class dependencies',
        action='store_true'
    )
    parser.add_argument(
        '--input-formats',
        help='Display class input formats',
        action='store_true'
    )
    parser.add_argument(
        '--output-formats',
        help='Display output formats',
        action='store_true'
    )
    parser.add_argument(
        '--argument-formats',
        help='Display the argument specification',
        action='store_true'
    )

    args = parser.parse_args(argv[1:])

    args = {k: v for k, v in vars(args).items() if v is not None}

    # if no flags are given, set all of them to True (display everything)
    if not any([v for v in args.values() if type(v) is bool]):
        for k, v in args.items():
            args[k] = True if type(v) is bool else v

    resulting_output = generate_class_info(**args)

    for result_line in resulting_output:
        print(result_line, end='')
        pass


if __name__ == '__main__':
    main(sys.argv)
