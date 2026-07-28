# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A set of functions for creating and manipulating Python dictionaries with
Kenning block configuration, parsed from various sources.

Making any changes to standard config dicts directly, without using any of
these functions, is strongly discouraged.
"""

import argparse
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

KenningBlockConfigDict = Dict[str, Dict[str, Any]]

# Names for modules/directories that store Kenning blocks of various types.
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
INFERENCE_LOOPS = "inferenceloops"
RUNTIME_PROTOCOLS = "protocols"
RUNTIMES = "runtimes"
AUTOML = "automl"
REPORT = "report"
CONVERTERS = "converters"


class ConfigKey(str, Enum):
    """
    Enum with available block types.

    `name` property defines block type name in YAML/JSON file config.
    `value` property defines Kenning module in which the block type
        implementations are.
    """

    dataset = DATASETS
    runtime = RUNTIMES
    optimizers = OPTIMIZERS
    platform = PLATFORMS
    protocol = RUNTIME_PROTOCOLS
    model_wrapper = MODEL_WRAPPERS
    runtime_builder = RUNTIME_BUILDERS
    inference_loop = INFERENCE_LOOPS
    dataconverter = DATA_CONVERTERS
    automl = AUTOML
    report = REPORT


# Keys used in the config dict (see this file's docstring for dict structure).
BLOCK_CONFIGURATIONS_KEY = "blocks"
UNAFFILIATED_PARAMETERS_KEY = "free_flags"
BLOCK_CONFIG_PARAMETERS_KEY = "parameters"
BLOCK_DIRECT_ARGUMENTS_KEY = "direct_extra_scenario_arguments"

# A block for a given block type can be defined both in JSON/YAML config, and
# in CLI, using a special flag. This dict maps block types to their CLI flags.
CONFIG_KEY_TO_CLS_FLAG = {
    ConfigKey.dataset: "--dataset-cls",
    ConfigKey.runtime: "--runtime-cls",
    ConfigKey.optimizers: "--compiler-cls",
    ConfigKey.platform: "--platform-cls",
    ConfigKey.protocol: "--protocol-cls",
    ConfigKey.model_wrapper: "--modelwrapper-cls",
    ConfigKey.runtime_builder: "--runtimebuilder-cls",
    ConfigKey.inference_loop: "--inferenceloop-cls",
    ConfigKey.dataconverter: "--dataconverter-cls",
    ConfigKey.automl: "--automl-cls",
    ConfigKey.report: "--report-cls",
}


def from_flag_to_name(s: str) -> str:
    """
    Converts argparse flag name to snake-case name (as used in
    argparse.Namespace attributes).

    Parameters
    ----------
    s:str
        Argparse flag name (example: '--sample-flag-name').

    Returns
    -------
    str
        Attribute name in argparse.Namespace corresponding to the flag
        (example: 'sample_flag_name').
    """
    return s.lstrip("-").replace("-", "_")


def module_path_to_class_name(path: Optional[str]) -> Optional[str]:
    """
    Extracts class name from a full Python module path (example:
    'some.pth.module.ExampleClass' to 'ExampleClass').

    Parameters
    ----------
    path: Optional[str]
        Either None, a class name, or Python path to a class.

    Returns
    -------
    Optional[str]
        Returns None for None, and a class name otherwise.
    """
    return path.rsplit(".", 1)[-1] if path else None


def argparse_to_config_dict(
    args: argparse.Namespace
) -> KenningBlockConfigDict:
    """
    Builds a config dict from an argparse.Namespace, containing
    parsed CLI flags. All flags that define Kenning blocks (those listed in
    CONFIG_KEY_TO_CLS_FLAG dict above) are recognized, and those blocks are
    placed in the block config (under BLOCK_CONFIGURATIONS_KEY). All other
    parsed flags are placed in the UNAFFILIATED_PARAMETERS_KEY section.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace containing parsed flags.

    Returns
    -------
    KenningBlockConfigDict
        A config dict built from scratch, based on the argparse.
    """
    result = {BLOCK_CONFIGURATIONS_KEY: {}, UNAFFILIATED_PARAMETERS_KEY: {}}
    args_attributes = vars(args).keys()
    for attribute in args_attributes:
        is_block_class_name = False
        for key, value in CONFIG_KEY_TO_CLS_FLAG.items():
            if attribute == from_flag_to_name(value):
                result[BLOCK_CONFIGURATIONS_KEY][key.value] = {
                    module_path_to_class_name(getattr(args, attribute)): {
                        BLOCK_CONFIG_PARAMETERS_KEY: {},
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    }
                }
                is_block_class_name = True
        if not is_block_class_name:
            result[UNAFFILIATED_PARAMETERS_KEY][attribute] = getattr(
                args, attribute
            )
    return result


def yaml_or_json_to_config_dict(config: Dict) -> KenningBlockConfigDict:
    """
    Builds a full config dict from a dict returned by a YAML/JSON
    parser, representing a configuration file.

    Parameters
    ----------
    config: Dict
        Python dict from a valid parsed YAML/JSON.

    Returns
    -------
    KenningBlockConfigDict
        A config dict built from scratch, based on the file config.
    """
    result = {BLOCK_CONFIGURATIONS_KEY: {}, UNAFFILIATED_PARAMETERS_KEY: {}}
    for block_type, elements in config.items():
        block_type = getattr(ConfigKey, block_type)
        blocks = []
        # There can be multiple optimizers in a config, but all other blocks
        # can have only 1 implementation active at a time.
        if block_type == ConfigKey.optimizers:
            blocks = elements
        else:
            blocks = [elements]
        final_block_configs = {}
        for block in blocks:
            block_key = (
                module_path_to_class_name(block["type"])
                if "type" in block
                else None
            )
            block_parameters = (
                block["parameters"] if "parameters" in block else {}
            )
            final_block_configs[block_key] = {
                BLOCK_CONFIG_PARAMETERS_KEY: block_parameters,
                BLOCK_DIRECT_ARGUMENTS_KEY: {},
            }
            result[BLOCK_CONFIGURATIONS_KEY][
                block_type.value
            ] = final_block_configs
    return result


def merge_config_dicts(
    *dicts: List[KenningBlockConfigDict]
) -> KenningBlockConfigDict:
    """
    Merges multiple config dicts.

    Parameters
    ----------
    *dicts: List[KenningBlockConfigDict]
        Multiple Kenning configuration dictionaries. Latter dictionaries
        overwrite the former ones, if both contain a parameter with the same
        name. Example: when calling merge_config_dicts(A, B, C), parameters
        in B will overwrite same-name parameters in A, while C will overwrite
        both A and B.

    Returns
    -------
    KenningBlockConfigDict
        Single merged dictionary.
    """
    result = dicts[0]
    dicts = dicts[1:]
    for dict in dicts:
        result[UNAFFILIATED_PARAMETERS_KEY] = (
            result[UNAFFILIATED_PARAMETERS_KEY]
            | dict[UNAFFILIATED_PARAMETERS_KEY]
        )
        for block_type, blocks in dict[BLOCK_CONFIGURATIONS_KEY].items():
            if block_type not in result[BLOCK_CONFIGURATIONS_KEY]:
                result[BLOCK_CONFIGURATIONS_KEY][block_type] = blocks
                continue
            for class_name, parameters in blocks.items():
                if (
                    class_name
                    not in result[BLOCK_CONFIGURATIONS_KEY][block_type]
                ):
                    result[BLOCK_CONFIGURATIONS_KEY][block_type][
                        class_name
                    ] = parameters
                    continue
                for parameter_type in [
                    BLOCK_CONFIG_PARAMETERS_KEY,
                    BLOCK_DIRECT_ARGUMENTS_KEY,
                ]:
                    result[BLOCK_CONFIGURATIONS_KEY][block_type][class_name][
                        parameter_type
                    ] = (
                        result[BLOCK_CONFIGURATIONS_KEY][block_type][
                            class_name
                        ][parameter_type]
                        | parameters[parameter_type]
                    )
    return result


def filter_block_types_from_config_dict(
    block_types: List[ConfigKey], config_dict: KenningBlockConfigDict
) -> KenningBlockConfigDict:
    """
    Removes from a config dict all block types that are not in the provided
    list.

    Parameters
    ----------
    block_types: List[ConfigKey]
        Accepted block types.
    config_dict: KenningBlockConfigDict
        Dictionary to change (NOTE: function operates directly on the dict,
        without making a copy).

    Returns
    -------
    KenningBlockConfigDict
        Changed dictionary, with all block types that are not in block_types
        removed.
    """
    block_types_to_delete = []
    for block_type in config_dict[BLOCK_CONFIGURATIONS_KEY]:
        if ConfigKey(block_type) not in block_types:
            block_types_to_delete.append(block_type)
    block_configurations = config_dict[BLOCK_CONFIGURATIONS_KEY]
    for block_type in block_types_to_delete:
        del block_configurations[block_type]
    config_dict[BLOCK_CONFIGURATIONS_KEY] = block_configurations
    return config_dict


def apply_default_blocks_by_block_type(
    defaults: Dict[ConfigKey, str], config_dict: KenningBlockConfigDict
) -> KenningBlockConfigDict:
    """
    In select block types that are either not in the config dict, or contain
    no blocks, or contain a None block, this function will set the provided
    default block.

    Parameters
    ----------
    defaults: Dict[ConfigKey, str]
        A Python dictionary with default block names for various block types.
    config_dict: KenningBlockConfigDict
        Dictionary to change (NOTE: function operates directly on the dict,
        without making a copy).

    Returns
    -------
    KenningBlockConfigDict
        Changed dictionary, with the defaults applied where needed.
    """
    for block_type, blocks in config_dict[BLOCK_CONFIGURATIONS_KEY].items():
        if block_type not in defaults:
            continue
        default_block = defaults[block_type]
        if len(blocks) == 0:
            blocks = {
                default_block: {
                    BLOCK_CONFIG_PARAMETERS_KEY: {},
                    BLOCK_DIRECT_ARGUMENTS_KEY: {},
                }
            }
        else:
            if None in blocks:
                blocks[default_block] = blocks[None]
                del blocks[None]
    return config_dict


def set_block_direct_argument(
    argument_name: str,
    argument_value: Any,
    config: KenningBlockConfigDict,
    block_type: ConfigKey,
    block: Optional[str] = None,
) -> KenningBlockConfigDict:
    """
    Places the given argument in the given block's configuration, under the
    BLOCK_DIRECT_ARGUMENTS_KEY. If an argument with the same name already
    exists, it will be overwritten.

    Parameters
    ----------
    argument_name: str
        Name under which the argument will be saved. Must match exactly the
        name of the constructor argument of the block's class that is supposed
        to be assigned the value.
    argument_value: Any
        Value of the argument that will be passed to the block.
    config: KenningBlockConfigDict
        Config dict to apply the change to (NOTE: function operates directly
        on the dict, without making a copy).
    block_type: ConfigKey
        Block type, for which the argument is to be applied. If there is no
        such block type in the dict, no changes will be made.
    block: Optional[str]
        Name of the specific block, for which the argument is. If set to None,
        the argument will be assigned to all blocks in the given block type.If
        there is no such block in the dict, no changes will be made.

    Returns
    -------
    KenningBlockConfigDict
        Changed config dict.
    """
    if block_type not in config[BLOCK_CONFIGURATIONS_KEY]:
        return config
    if block:
        if block not in config[BLOCK_CONFIGURATIONS_KEY][block_type]:
            return config
        config[BLOCK_CONFIGURATIONS_KEY][block_type][block][
            BLOCK_DIRECT_ARGUMENTS_KEY
        ][argument_name] = argument_value
        return config
    for block_parameters in config[BLOCK_CONFIGURATIONS_KEY][
        block_type
    ].values():
        block_parameters[BLOCK_DIRECT_ARGUMENTS_KEY][
            argument_name
        ] = argument_value
    return config
