# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Kenning block dependency solver. This module identifies which kenning blocks
will have the role of server and client in their communication.
"""

import inspect
from typing import Dict, List, Type, get_args, get_origin

from kenning.core.exceptions import ConfigurationError
from kenning.utils.class_loader import get_base_classes_dict, load_class


def _is_type_kenning_block(cls: Type):
    core_classes = tuple(v[1] for _, v in get_base_classes_dict().items())
    try:
        return issubclass(cls, core_classes)
    except TypeError:
        # Not a Kenning base class
        return False


class DependencySolver:
    """
    Solves and maps dependency relationships between Kenning blocks.
    """

    def __init__(self, block_configs: Dict):
        """
        Initializes the dependency solver.

        Parameters
        ----------
        block_configs : Dict
            Dictionary containing configurations of blocks to run.

        Raises
        ------
        ConfigurationError
            Raised when a passed block is not a name (string)
            or the class type.
        """
        self.blocks = []
        # Iterate and save each block into a flat list.
        for _, blocks in block_configs.items():
            for block_name in blocks:
                if isinstance(block_name, str):
                    self.blocks.append(load_class(block_name))
                elif isinstance(block_name, Type):
                    self.blocks.append(block_name)
                else:
                    raise ConfigurationError(
                        "Kenning block should either be a name or its class"
                    )

    def get_dependency_graph(self) -> Dict:
        """
        Generates a directed acyclic graph mapping clients to
        their server dependencies.

        Returns
        -------
        Dict
            A dictionary where keys are client block names and
            values are lists of server block names.
        """
        graph = {}
        for block in self.blocks:
            graph[block.__name__] = self._get_class_dependencies(block)

        return graph

    def _identify_dep_types(self, cls: Type) -> List[Type]:
        """
        Identifies Kenning block types required by a given class constructor.

        Parameters
        ----------
        cls : Type
            The class to inspect.

        Returns
        -------
        List[Type]
            A list of dependency types required by the class.
        """
        sig = inspect.signature(cls)
        dep_types = []
        for name, param in sig.parameters.items():
            # Loop through all the parameters in the __init__
            # function of the class
            if param.annotation == inspect.Parameter.empty:
                continue

            types_to_check = [param.annotation]
            has_kenning_block = False

            while types_to_check:
                curr_type = types_to_check.pop(0)
                origin = get_origin(curr_type)

                if origin is not None:
                    # It is a typing construct like
                    # Union, Optional, or List
                    types_to_check.extend(get_args(curr_type))
                elif _is_type_kenning_block(curr_type):
                    has_kenning_block = True
                    break

            if has_kenning_block:
                dep_types.append(param.annotation)
        return dep_types

    def _append_matching_dependencies(
        self, target_type: Type, client_dependencies: List[str]
    ):
        """
        Appends block names that match the target dependency
        type to the client's dependency list.

        Parameters
        ----------
        target_type : Type
            The target dependency class type to match against active blocks.
        client_dependencies : List[str]
            The list of resolved string dependencies for the client block.
        """
        for active_block in self.blocks:
            if (
                inspect.isclass(target_type)
                and issubclass(active_block, target_type)
                and active_block.__name__ not in client_dependencies
            ):
                client_dependencies.append(active_block.__name__)

    def _get_class_dependencies(self, block: Type) -> List[str]:
        """
        Retrieves the resolved string dependencies for a specific block class.

        Parameters
        ----------
        block : Type
            The block class to evaluate.

        Returns
        -------
        List[str]
            A list of server block names that the client depends on.
        """
        client_dependencies = []
        dep_types = self._identify_dep_types(block)

        # Ensure that each block has the required
        # dependencies mapped by their string names
        for dep_type in dep_types:
            origin = get_origin(dep_type)
            if origin is not None:
                for kenning_type in get_args(dep_type):
                    self._append_matching_dependencies(
                        kenning_type, client_dependencies
                    )
            else:
                self._append_matching_dependencies(
                    dep_type, client_dependencies
                )

        return client_dependencies
