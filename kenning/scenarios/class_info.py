#!/usr/bin/env python

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
A script that provides information about a given class provided in the argument.

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

KEYWORDS = ['inputtypes', 'outputtypes', 'arguments_structure']


def print_class_module_docstrings(syntax_node: Union[ast.ClassDef, ast.Module]):
    """
    Prints formatted docstrings of modules and classes
    """

    docstring = ast.get_docstring(syntax_node, clean=True)

    if not docstring:
        return

    docstring = '\n'.join(
        ['\t' + docstr for docstr in docstring.strip('\n').split('\n')])

    if isinstance(syntax_node, ast.ClassDef):
        print(f'Class: {syntax_node.name}\n')
        print(f'{docstring}')

    if isinstance(syntax_node, ast.Module):
        print('Module description:\n')
        print(docstring)

    print('')


def get_dependency(syntax_node: Union[ast.Import, ast.ImportFrom]) \
        -> str:
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

        if 'kenning' in dependency_path:
            # TODO recursively check for dependencies in kenning modules
            return ''

        try:
            importlib.import_module(module_path)
            return '* ' + dependency_path
        except ImportError or ModuleNotFoundError as e:
            return f'* {dependency_path} - Not available (Reason: {e})'


def print_input_specification(syntax_node: ast.Assign):
    for input_format in syntax_node.value.keys:
        print(f'* {input_format.value}')
    print('')


def print_output_specification(syntax_node: ast.Assign):
    for output_format in syntax_node.value.elts:
        print(f'* {output_format.value}')
    print('')


def print_arguments_structure(syntax_node: ast.Assign):
    for argument, argument_specification_dict in zip(syntax_node.value.keys,
                                                     syntax_node.value.values):
        print(f'* {argument.value}')

        for key, value in zip(argument_specification_dict.keys,
                              argument_specification_dict.values):
            key_str = astunparse.unparse(key).strip().\
                removeprefix("'").removesuffix("'")
            value_str = astunparse.unparse(value).strip().\
                removeprefix("'").removesuffix("'")

            print(f'  * {key_str}: {value_str}')


def generate_class_info(target: str):
    """
    Wrapper function that displays information about a given class

    Parameters
    ----------


    """

    target_path = target
    if not target.endswith('.py'):
        target_path = target.replace('.', '/')
        target_path += '.py'

    if not os.path.exists(target_path):
        print('This class does not exist')
        return

    with open(target_path) as file:
        parsed_file = ast.parse(file.read())

    syntax_nodes = ast.walk(parsed_file)

    class_nodes = []
    dependency_nodes = []

    input_specification_node = None
    output_specification_node = None
    arguments_structure_node = None

    for node in syntax_nodes:
        if isinstance(node, (ast.ClassDef, ast.Module)):
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

    for node in class_nodes:
        print_class_module_docstrings(node)

    print('Dependencies:')
    dependencies: List[str] = []
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    for node in dependency_nodes:
        dependency_str = get_dependency(node)
        if dependency_str == '':
            continue
        dependencies.append(dependency_str)

    [print(dep_str) for dep_str in list(set(dependencies))]

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    print('')

    print("Input formats:")
    if input_specification_node:
        print_input_specification(input_specification_node)

    print("Output formats:")
    if output_specification_node:
        print_output_specification(output_specification_node)

    print("Arguments specification:")
    if arguments_structure_node:
        print_arguments_structure(arguments_structure_node)


def main(argv):
    parser = argparse.ArgumentParser(argv[0])

    parser.add_argument(
        'target',
        help='',
        type=str
    )

    args = parser.parse_args(argv[1:])

    generate_class_info(target=args.target)


if __name__ == '__main__':
    main(sys.argv)
