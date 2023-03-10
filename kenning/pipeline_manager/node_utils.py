# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from kenning.pipeline_manager.core import Node
from kenning.utils.class_loader import load_class
from kenning.utils.logger import get_logger


_LOGGER = get_logger()


def add_node(
        node_list: Dict[str, Node],
        nodemodule: str,
        category: str,
        type: str):
    """
    Loads a class containing Kenning block and adds it to available nodes.

    If the class can't be imported due to import errors, it is not added.

    Parameters
    ----------
    node_list: List[Node]
        List of nodes to add to the specification
    nodemodule : str
        Python-like path to the class holding a block to add to specification
    category : str
        Category of the block
    type : str
        Type of the block added to the specification
    """
    try:
        nodeclass = load_class(nodemodule)
        node_list[nodeclass.__name__] = (
            Node(nodeclass.__name__, category, type, nodeclass)
        )
    except (ModuleNotFoundError, ImportError, Exception) as err:
        msg = f'Could not add {nodemodule}. Reason:'
        _LOGGER.warn('-' * len(msg))
        _LOGGER.warn(msg)
        _LOGGER.warn(err)
        _LOGGER.warn('-' * len(msg))


def get_category_name(kenning_class):
    """
    Turns 'kenning.module.submodule1.submodule2. ... .specific_module'
    into 'module (submodule1, submodule2, ...)'
    """
    names = kenning_class.__module__.split(".")[1:-1]
    base_module, *submodules = names
    submodules = ', '.join(submodules)
    base_module = str.capitalize(base_module)
    if submodules == '':
        return base_module
    return f"{base_module} ({submodules})"
