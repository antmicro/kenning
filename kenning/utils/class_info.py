# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A module with methods for collecting details on Kenning classes.
"""

import ast
import importlib
import inspect
import os.path
from importlib.util import find_spec
from typing import Dict, List, Optional, Tuple, Type, Union

import astunparse
from isort import place_module
from jsonschema.exceptions import ValidationError
from rst_to_myst.mdformat_render import rst_to_myst

from kenning.core.dataprovider import DataProvider
from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.core.outputcollector import OutputCollector
from kenning.core.protocol import Protocol
from kenning.core.runner import Runner
from kenning.core.runtime import Runtime
from kenning.utils.args_manager import (
    from_argparse_name,
    jsontype_to_type,
    to_argparse_name,
)
from kenning.utils.excepthook import (
    MissingKenningDependencies,
    find_missing_optional_dependency,
)

KEYWORDS = ["inputtypes", "outputtypes", "arguments_structure"]


class Argument:
    """
    Class representing an argument. Fields that are empty are not displayed.
    """

    def __init__(self):
        self.name = ""
        self.argparse_name = ""
        self.description = ""
        self.required = ""
        self.default = ""
        self.nullable = ""
        self.type = ""
        self.enum: List[str] = []

    def __repr__(self):
        lines = [f"* `{self.name}`"]

        if self.description:
            lines.append(f"* description: {self.description}")
        if self.argparse_name:
            lines.append(f"* argparse flag: `{self.argparse_name}``")
        if self.type:
            lines.append(f"* type: `{self.type}`")
        if self.required:
            lines.append(f"* required: `{self.required}`")
        if self.default:
            lines.append(f"* default value: `{self.default}`")
        if self.nullable:
            lines.append(f"* Can be undefined: `{self.nullable}`")

        if len(self.enum) != 0:
            lines.append("* `allowed values`:")
        for element in self.enum:
            lines.append(f"    * `{element}`")

        return "\n".join(lines)


class ClassInfoInvalidArgument(Exception):
    """
    Exception raised when the arguments provided are not valid.
    """

    pass


def get_class_module_name(syntax_node: Union[ast.ClassDef, ast.Module]) -> str:
    """
    Displays class name from syntax node.

    Parameters
    ----------
    syntax_node: Union[ast.ClassDef, ast.Module]
        Class syntax node

    Returns
    -------
    str
        Formatted Markdown-like string to be printed later.
    """
    if isinstance(syntax_node, ast.ClassDef):
        return f"# {syntax_node.name}\n\n"


def get_class_module_docstrings(
    syntax_node: Union[ast.ClassDef, ast.Module]
) -> str:
    """
    Displays docstrings of provided class or module.

    Parameters
    ----------
    syntax_node: Union[ast.ClassDef, ast.Module]
        Syntax node representing a class or module

    Returns
    -------
    str
        Formatted Markdown-like string to be printed later.
    """
    docstring = ast.get_docstring(syntax_node, clean=True)

    if not docstring:
        return f"# Class {syntax_node.name}\n\n"

    docstring = rst_to_myst(docstring).text

    if isinstance(syntax_node, ast.ClassDef):
        return f"# Class {syntax_node.name}\n\n{docstring}\n\n"

    if isinstance(syntax_node, ast.Module):
        return f"# Module description\n{docstring}\n"


def get_dependency(syntax_node: Union[ast.Import, ast.ImportFrom]) -> str:
    """
    Extracts a dependency from an import syntax node and checks whether the
    dependency is satisfied. It also skips internal kenning modules.

    Parameters
    ----------
    syntax_node: Union[ast.Import, ast.ImportFrom]
        An assignment like `from iree.compiler import version``

    Returns
    -------
    str
        Formatted Markdown-like string to be printed later. Empty strings
        represent dependencies that were skipped - either they belong to
        Kenning or are provided by the default python distribution
    """
    for dependency in syntax_node.names:
        module_path = ""
        dependency_path = ""
        if isinstance(syntax_node, ast.ImportFrom):
            dependency_path = f"{syntax_node.module}.{dependency.name}"
            module_path = f"{syntax_node.module}"

        if isinstance(syntax_node, ast.Import):
            dependency_path = f"{dependency.name}"
            module_path = dependency_path

        if module_path == "" or dependency_path == "":
            return ""

        try:
            importlib.import_module(module_path)

            if "kenning" in dependency_path:
                return ""

            if place_module(module_path) == "STDLIB":
                return ""

            return f"* `{dependency_path}`\n"
        except (ImportError, ModuleNotFoundError, Exception) as e:
            if (
                not hasattr(e, "name")
                or e.name is None
                or find_missing_optional_dependency(e.name) is None
            ):
                return f"* `{dependency_path}` - Not available (Reason: {e})\n"

            err = MissingKenningDependencies(
                name=module_path,
                path=dependency_path,
                optional_dependencies=find_missing_optional_dependency(e.name),
            )
            return f"* `{dependency_path}` - Not available (Reason: {e})\n    {err}"  # noqa: E501


def get_input_specification(syntax_node: ast.Assign) -> str:
    """
    Displays information about the input specification as bullet points.

    Parameters
    ----------
    syntax_node: ast.Assign
        An assignment like `inputtypes = []`

    Returns
    -------
    str
        Formatted Markdown-like string to be printed later.
    """
    input_formats = ""

    if (
        isinstance(syntax_node.value, ast.List)
        and len(syntax_node.value.elts) == 0
    ):
        return ""

    if isinstance(syntax_node.value, ast.List):
        for input_format in syntax_node.value.elts:
            input_formats += f"* `{input_format.value}`\n"
        return input_formats

    for input_format in syntax_node.value.keys:
        input_formats += f"* `{input_format.value}`\n"

    return input_formats


def parse_io_dict_node_to_string(dict_node: ast.Dict) -> List[str]:
    """
    Parses an ast.Dict to a nicely formatted list of strings in Markdown
    format.

    Parameters
    ----------
    dict_node: ast.Dict
        AST dict node to extract the data from

    Returns
    -------
    List[str]
        List of formatted Markdown-like strings to be printed later.
    """
    # formatted lines to be returned
    resulting_output = []

    if not isinstance(dict_node, ast.Dict):
        return [""]

    for key, value in zip(dict_node.keys, dict_node.values):
        if not isinstance(value, ast.List):
            resulting_output.append(f"* `{key.value}`: `{value.value}`")
            continue
        dict_elements = value.elts

        if len(dict_elements) > 0:
            resulting_output.append(f"### {key.value}\n\n")

        for dict_element in dict_elements:
            if isinstance(dict_element.values[0], ast.Constant):
                resulting_output.append(
                    f"* `{dict_element.values[0].value}`\n"
                )
            else:
                # if the first value (name) is not a string,
                # use the variable name
                resulting_output.append(f"* `{dict_element.values[0].id}`\n")

            for key, value in zip(
                dict_element.keys[1:], dict_element.values[1:]
            ):
                resulting_output.append(
                    f"    * `{key.value}`: "
                    f"`{clean_variable_name(value)}`\n"
                )

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
    List[str]
        List of formatted Markdown-like strings to be printed later.
    """
    io_spec_function_node = None

    if not class_node.body:
        return []

    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef):
            continue

        if node.name != "_get_io_specification":
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

    return parse_io_dict_node_to_string(io_spec_dict_node)


def get_output_specification(syntax_node: ast.Assign) -> str:
    """
    Displays information about the output specification as bullet points.

    Parameters
    ----------
    syntax_node: ast.Assign
        An assignment like `outputtypes = ['iree']`

    Returns
    -------
    str
        Formatted Markdown-like string to be printed later.
    """
    for output_format in syntax_node.value.elts:
        return f"* `{output_format.value}`\n"


def clean_variable_name(variable_name: ast.AST) -> str:
    """
    Unparses and cleans a parsed variable name as string from single quotation
    marks and trailing whitespaces.

    Parameters
    ----------
    variable_name: ast.AST
        Variable to be cleaned up, e.g. "'tflite' "

    Returns
    -------
    str
        Cleaned up variable
    """
    return (
        astunparse.unparse(variable_name)
        .strip()
        .removeprefix("'")
        .removesuffix("'")
    )


def get_arguments_structure(syntax_node: ast.Assign, source_path: str) -> str:
    """
    Displays information about the argument structure specification as
    bullet points.

    Parameters
    ----------
    syntax_node: ast.Assign
        An assignment like `arguments_structure = {'compiler_args': {}}`
    source_path: str
        Source path of the code to be parsed

    Returns
    -------
    str
        Formatted Markdown-like string to be printed later.
    """
    output_string = ""

    for argument, argument_specification_dict in zip(
        syntax_node.value.keys, syntax_node.value.values
    ):
        argument_object = Argument()

        argument_object.name = argument.value

        for key, value in zip(
            argument_specification_dict.keys,
            argument_specification_dict.values,
        ):
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "list"
            ):
                argument_list_variable = (
                    astunparse.unparse(value)
                    .strip()
                    .removeprefix("'")
                    .removesuffix("'")
                    .replace("list(", "")
                    .replace(".keys())", "")
                )

                argument_keys, argument_type = evaluate_argument_list_of_keys(
                    argument_list_variable, source_path
                )

                argument_object.enum = argument_keys
                argument_object.type = argument_type
            elif isinstance(value, ast.Call) and isinstance(
                value.func, ast.Attribute
            ):
                key_str = clean_variable_name(key)
                value_str = clean_variable_name(value)

                argument_object.__setattr__(key_str, [value_str])

            elif key.value == "enum":
                argument_list_variable = clean_variable_name(value)

                enum_list, argument_type = evaluate_argument_list(
                    argument_list_variable, source_path
                )

                argument_object.enum = enum_list
                argument_object.type = argument_type

                if argument_type == "Error":
                    return f"*Error*: `{enum_list[0]}`"

            else:
                key_str = clean_variable_name(key)
                value_str = clean_variable_name(value)

                argument_object.__setattr__(key_str, value_str)

        output_string += argument_object.__repr__() + "\n"

    return output_string


def evaluate_argument_list_of_keys(
    argument_list_name: str, source_path: str
) -> Tuple[List[str], str]:
    """
    Evaluate an expression like `list(some_dict.keys())` and return the list
    of elements as strings.

    Parameters
    ----------
    argument_list_name: str
        Variable name that the list of keys is assigned to
    source_path: str
        Path of the code to be parsed

    Returns
    -------
    Tuple[List[str], str]
        tuple with the first argument being the list of
        evaluated elements and the second being the type as a string
    """
    with open(source_path, "r") as file:
        parsed_file = ast.parse(file.read())

    syntax_nodes = ast.walk(parsed_file)

    argument_list_keys = []
    argument_type = ""

    for node in syntax_nodes:
        if not isinstance(node, ast.Assign):
            continue

        if not isinstance(node.targets[0], ast.Name):
            continue

        if not node.targets[0].id == argument_list_name:
            continue

        for key in node.value.keys:
            argument_list_keys.append(key.value)

        argument_type = f"List[{type(node.value.keys[0].value).__name__}]"

        break

    return argument_list_keys, argument_type


def evaluate_argument_list(
    argument_list_name: str, source_path: str
) -> Tuple[List[str], str]:
    """
    Evaluate an expression like `list('tflite', 'tvm')` and return the list
    of elements as strings.

    Parameters
    ----------
    argument_list_name: str
        Variable name that the list of elements is assigned to.
    source_path: str
        Path of the code to be parsed

    Returns
    -------
    Tuple[List[str], str]
        tuple with the first argument being the list of
        evaluated elements and the second being the type as a string
    """
    with open(source_path) as file:
        parsed_file = ast.parse(file.read())

    syntax_nodes = ast.walk(parsed_file)

    enum_elements = []
    argument_type = ""

    # argument list is an explicit python list (['int8', 'float16'])
    if argument_list_name.endswith("]") and argument_list_name[0] == "[":
        try:
            enum_elements = eval(argument_list_name)
        except NameError:
            return [
                "Static code analysis failed here, please import the "
                "necessary modules and/or try to load the class with "
                "arguments"
            ], "Error"

        if len(enum_elements) > 0:
            argument_type = f"List[{type(enum_elements[0]).__name__}]"
        else:
            argument_type = "List[]"
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

        argument_type = f"List[{type(node.value.elts[0].value).__name__}]"
        break

    return enum_elements, argument_type


def get_args_structure_from_parameterschema(
    parameterschema: Dict
) -> List[str]:
    """
    Returns argument structure in the form of Markdown-like strings based on
    the provided parameterschema.

    Parameters
    ----------
    parameterschema: Dict
        Arguments specification of the class

    Returns
    -------
    List[str]
        Formatted Markdown-like string to be printed later.
    """
    resulting_lines = []

    args_structure = parameterschema["properties"]

    required_args: List = []
    if "required" in parameterschema:
        required_args = parameterschema["required"]

    if not args_structure:
        return [""]

    for arg_name, arg_dict in args_structure.items():
        resulting_lines.append(f"### `{arg_name}`\n")

        if "description" in arg_dict:
            resulting_lines.append(f"* {arg_dict['description']}\n")

        resulting_lines.append(
            f"* argparse flag: `{to_argparse_name(arg_name)}`\n"
        )

        for key, value in arg_dict.items():
            # skip real_name as it is the same as arg_name
            # skip description, as it has been already handled
            if key in ["real_name", "description"]:
                continue

            # expand enums (lists)
            if isinstance(value, list):
                if len(value) == 1:
                    resulting_lines.append(f"* {key}: `{value[0]}`\n")
                    continue
                resulting_lines.append(f"* {key}\n")
                for elt in value:
                    resulting_lines.append(f"    * `{elt}`\n")
                continue

            # extract qualified class name if value is a class
            if inspect.isclass(value):
                resulting_lines.append(
                    f"* {key}: `{value.__module__}.{value.__qualname__}`\n"
                )
                continue

            resulting_lines.append(f"* {key}: `{value}`\n")

        if arg_name in required_args:
            resulting_lines.append("* required: `True`\n")

    return resulting_lines


def parse_io_spec_dict_to_str(dictionary: Dict) -> List[str]:
    """
    Recursively parses a dictionary to a list of formatted,
    Markdown-like strings.

    Parameters
    ----------
    dictionary: Dict
        A python dictionary to be parsed

    Return
    ------
    List[str]
        A list of formatted, Markdown-like strings
    """
    resulting_output = []

    dict_elements = []

    if not isinstance(dictionary, dict):
        return [""]

    for key, value in dictionary.items():
        resulting_output.append(f"### {key}\n\n")
        if not isinstance(value, list):
            resulting_output.append(f"* `{key}`: `{value}`\n")

        [dict_elements.append(elt) for elt in value]

    for dict_element in dict_elements:
        resulting_output.append(f'* `{dict_element["name"]}`\n')
        dict_element.pop("name", None)

        for key, value in dict_element.items():
            if isinstance(value, list):
                resulting_output.append(f"    * `{key}`\n")
                for elt in value:
                    resulting_output.append(f"        * `{elt}`\n")
                continue

            resulting_output.append(f"    * `{key}`: `{value}`\n")

    return resulting_output


def instantiate_object(
    imported_class: Type, parameterschema: Dict = {}, arguments: List[str] = []
) -> object:
    """
    Parses provided arguments into a dictionary, then creates an instance of
    the provided class.

    Parameters
    ----------
    imported_class: Type
        Class to create an instance of.
    parameterschema: Dict
        Argument structure of the class
    arguments: List[str]
        Arguments provided by the user, will be used to create an object

    Returns
    -------
    object
        An instance of imported_class

    Raises
    ------
    ClassInfoInvalidArgument:
        Raised when invalid name is provided for a class.
    """
    # create a dict of arguments that will be used to create an instance
    parsed_args: Dict = {}

    # split the arguments into lists with two elements, i.e. argparse_name and value # noqa: E501
    arg_tuples = [arguments[i : i + 2] for i in range(0, len(arguments), 2)]

    for arg_tuple in arg_tuples:
        argparse_name = from_argparse_name(arg_tuple[0])

        if argparse_name not in parameterschema["properties"].keys():
            raise ClassInfoInvalidArgument(
                f"Argparse name {to_argparse_name(argparse_name)} not present "
                f"in argument specification"
            )

        parameter = parameterschema["properties"][argparse_name]

        argument_type = str
        if "type" in parameter.keys() and isinstance(parameter["type"], list):
            argument_type = jsontype_to_type[parameter["type"][0]]

        parsed_args[argparse_name] = argument_type(arg_tuple[1])

    class_object = instantiate_object_based_on_base_class(
        imported_class, parsed_args
    )

    return class_object


def instantiate_object_based_on_base_class(
    imported_class: Type, parsed_args: Dict
) -> object:
    """
    Creates an object of the provided class, based on the arguments provided
    in dictionary format and the class it inherits from. Raises a custom
    exception when there was an error.

    Parameters
    ----------
    imported_class: Type
        Class to create an instance of.
    parsed_args: Dict
        Arguments following the arguments_structure of the class.

    Returns
    -------
    object
        An instance of imported_class

    Raises
    ------
    ClassInfoInvalidArgument:
        Raised when class could not be created or arguments were missing.
    """
    try:
        # create an object based on its base class
        if issubclass(imported_class, Runtime):
            return imported_class.from_json(
                json_dict=parsed_args, protocol=Protocol()
            )

        if issubclass(imported_class, ModelWrapper):
            return imported_class.from_json(
                json_dict=parsed_args, dataset=None, from_file=False
            )

        if issubclass(imported_class, Dataset):
            return imported_class.from_json(parsed_args)

        if issubclass(imported_class, Optimizer):
            return imported_class.from_json(
                dataset=None, json_dict=parsed_args
            )

        if issubclass(imported_class, DataProvider):
            return imported_class.from_json(
                json_dict=parsed_args,
                inputs_sources={},
                inputs_specs={},
                outputs={},
            )

        if issubclass(imported_class, OutputCollector):
            return imported_class.from_json(
                json_dict=parsed_args,
                inputs_specs={},
                inputs_sources={},
                outputs={},
            )

        if issubclass(imported_class, Runner):
            # TODO Runner class is to be updated
            return None

    except ValidationError as e:
        reason = str(e).partition("\n")[0]
        raise ClassInfoInvalidArgument(
            f"Could not create a {imported_class.__name__} object. "
            f"You need to provide the required arguments.\n"
            f"Reason: {reason}\n"
        )
    except FileNotFoundError as e:
        raise ClassInfoInvalidArgument(
            f"Could not create a {imported_class.__name__} object.\n"
            f"Reason: {e}\n"
        )

    return None


def get_class_description(target: str, class_name: str) -> str:
    """
    Extracts short description for the given class.
    Returns empty string if no description is provided.

    Parameters
    ----------
    target: str
        Target module-like path e.g.
        `kenning.modelwrappers.object_detection.yolov4.ONNXYOLOV4`
    class_name: str
        Name of a specific class to display information about

    Returns
    -------
    str
        List of formatted, Markdown-like lines to be printed
    """
    # if target contains a class, split to path and class name
    split_target = target.split(".")
    if split_target[-1][0].isupper():
        class_name = split_target[-1]
        split_target = split_target[:-1]

    target = ".".join(split_target)

    target_path = find_spec(target).origin

    if not os.path.exists(target_path):
        return ""

    with open(target_path, "r") as file:
        parsed_file = ast.parse(file.read())

    syntax_nodes = ast.walk(parsed_file)
    for node in syntax_nodes:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name != class_name:
            continue
        docstring = ast.get_docstring(node, clean=True)
        if not docstring:
            return ""
        return docstring
    return ""


def generate_class_info(
    target: str,
    class_name: str = "",
    docstrings: bool = True,
    dependencies: bool = True,
    input_formats: bool = True,
    output_formats: bool = True,
    argument_formats: bool = True,
    load_class_with_args: Optional[List[str]] = None,
) -> List[str]:
    """
    Wrapper function that handles displaying information about a class.

    Parameters
    ----------
    target: str
        Target module-like path e.g.
        `kenning.modelwrappers.object_detection.yolov4.ONNXYOLOV4`
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
    load_class_with_args: Optional[List[str]]
        List of arguments provided to load a specific class with arguments

    Returns
    -------
    List[str]
        List of formatted, Markdown-like lines to be printed
    """
    resulting_lines = []

    # if target contains a class, split to path and class name
    split_target = target.split(".")
    if split_target[-1][0].isupper():
        class_name = split_target[-1]
        split_target = split_target[:-1]

    target = ".".join(split_target)

    target_path = find_spec(target).origin

    if not os.path.exists(target_path):
        return [f"File {target_path} does not exist\n"]

    with open(target_path, "r") as file:
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

    if class_name:
        # try to load the class into memory
        try:
            imported_class = getattr(
                importlib.import_module(find_spec(target).name), class_name
            )

            parameterschema = imported_class.form_parameterschema()

        except (ModuleNotFoundError, ImportError, Exception):
            resulting_lines.append(
                "Warning: Only static code analysis will be performed - "
                f"cannot import class {class_name}.\nTry installing the "
                f"required dependencies or loading the class with "
                f"arguments.\n\n"
            )

    # create an object when it has no required arguments if possible
    if imported_class and load_class_with_args is not None:
        try:
            class_object = instantiate_object(
                imported_class, parameterschema, load_class_with_args
            )
        except ClassInfoInvalidArgument as e:
            return [str(e)]

    # perform static code analysis
    for node in syntax_nodes:
        if isinstance(node, ast.ClassDef):
            class_nodes.append(node)

            io_specification = get_io_specification(node)
            if len(io_specification) > 0:
                io_specification_lines[node] = io_specification
                found_io_specification = True

        if isinstance(node, ast.Module) and class_name == "":
            class_nodes.append(node)

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            dependency_nodes.append(node)

        if isinstance(node, ast.Assign) and isinstance(
            node.targets[0], ast.Name
        ):
            if node.targets[0].id not in KEYWORDS:
                continue

            if node.targets[0].id == KEYWORDS[0]:
                input_specification_node = node
            if node.targets[0].id == KEYWORDS[1]:
                output_specification_node = node

            if node.targets[0].id == KEYWORDS[2]:
                arguments_structure_node = node

    # prepare output
    for node in class_nodes:
        if docstrings:
            if len(class_nodes) == 0:
                resulting_lines.append(
                    f"Class {class_name} has not been found"
                )
                return resulting_lines
            resulting_lines.append(get_class_module_docstrings(node))
        else:
            resulting_lines.append(get_class_module_name(node))

        if input_formats or output_formats:
            if imported_class and hasattr(
                class_object, "get_io_specification"
            ):
                # object has been created - detailed i/o specification found
                found_io_specification = True
                resulting_lines.append("## Block input/output formats\n\n")
                io_spec = class_object.get_io_specification()
                resulting_lines += parse_io_spec_dict_to_str(io_spec)
                resulting_lines.append("\n")

            elif node in io_specification_lines:
                # no object, but i/o specification found - extract statically
                # from source code
                resulting_lines.append("## Block input/output formats\n")
                resulting_lines += io_specification_lines[node]
                resulting_lines.append("\n")

    if dependencies:
        resulting_lines.append("## Dependencies\n\n")
        dependencies: List[str] = []
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        for node in dependency_nodes:
            dependency_str = get_dependency(node)
            if dependency_str == "":
                continue
            dependencies.append(dependency_str)

        for dep_str in list(set(dependencies)):
            resulting_lines.append(dep_str)

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        resulting_lines.append("\n")

    if input_formats or output_formats and not found_io_specification:
        resulting_lines.append("## Block input/output formats\n\n")
    if input_formats and not found_io_specification:
        resulting_lines.append("### Input formats\n")
        if input_specification_node:
            data = get_input_specification(input_specification_node)
            resulting_lines.append(data if data else "No inputs")
        else:
            resulting_lines.append("No inputs")
        resulting_lines.append("\n\n")

    if output_formats and not found_io_specification:
        resulting_lines.append("### Output formats\n")
        if output_specification_node:
            data = get_output_specification(output_specification_node)
            resulting_lines.append(data if data else "No outputs")
        else:
            resulting_lines.append("No outputs")
        resulting_lines.append("\n\n")

    if argument_formats:
        resulting_lines.append("## Arguments' specification\n\n")
        if parameterschema:
            resulting_lines += get_args_structure_from_parameterschema(
                parameterschema
            )

        elif arguments_structure_node:
            data = get_arguments_structure(
                arguments_structure_node, target_path
            )
            resulting_lines.append(data if data else "No arguments")
        else:
            resulting_lines.append("No arguments")
        resulting_lines.append("\n\n")

    return resulting_lines
