# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, NamedTuple, List, Tuple, Union

from kenning.utils.class_loader import load_class
from kenning.utils.logger import get_logger

_LOGGER = get_logger()


class Node(NamedTuple):
    """
    NamedTuple specifying the single node in specification.

    Attributes
    ----------
    name : str
        Name of the block
    category : str
        Category of the node (where it appears in the available blocks menu)
    type : str
        Type of the block (for internal usage in pipeline manager)
    cls : object
        Class object to extract parameters from.
    """
    name: str
    category: str
    type: str
    cls: object


def add_node(
        node_list: List,
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
        node_list.append(Node(nodeclass.__name__, category, type, nodeclass))
    except (ModuleNotFoundError, ImportError, Exception) as err:
        msg = f'Could not add {nodemodule}. Reason:'
        _LOGGER.warn('-' * len(msg))
        _LOGGER.warn(msg)
        _LOGGER.warn(err)
        _LOGGER.warn('-' * len(msg))


class BaseDataflowHandler:
    def __init__(
            self,
            nodes: List[Node],
            io_mapping: Dict
    ):
        """
        Base class for handling different types of kenning specification,
        converting to and from Pipeline Manager formats, and running them
        in Kenning.

        Parameters
        ----------
        nodes : List[Node]
            List of available nodes for this dataflow type
        io_mapping : Dict
            Mapping used by Pipeline Manager for defining the shape
            of each node type.
        """
        self.nodes = nodes
        self.io_mapping = io_mapping

    def get_specification(self) -> Dict:
        """
        Prepares core-based Kenning classes to be sent to Pipeline Manager.

        For every class in `nodes` it uses its parameterschema to create
        a corresponding dataflow specification.

        Returns
        -------
        Dict:
            Specification ready to be send to Pipeline Manager
        """
        specification = {
            'metadata': {},
            'nodes': []
        }

        def strip_io(io_list: list) -> list:
            """
            Strips every input/output from metadata and leaves only
            `name` and `type` keys.
            """
            return [
                {
                    'name': io['name'],
                    'type': io['type']
                }
                for io in io_list
            ]

        for node in self.nodes:
            parameterschema = node.cls.form_parameterschema()

            properties = []
            for name, props in parameterschema['properties'].items():
                new_property = {'name': name}

                if 'default' in props:
                    new_property['default'] = props['default']

                if 'description' in props:
                    new_property['description'] = props['description']

                # Case for an input with range defined
                if 'enum' in props:
                    new_property['type'] = 'select'
                    new_property['values'] = list(map(str, props['enum']))
                # Case for a single value input
                elif 'type' in props:
                    if 'array' in props['type']:
                        new_property['type'] = 'list'
                        if 'items' in props and 'type' in props['items']:
                            dtype = props['items']['type']
                            new_property['dtype'] = dtype
                    elif 'boolean' in props['type']:
                        new_property['type'] = 'checkbox'
                    elif 'string' in props['type']:
                        new_property['type'] = 'text'
                    elif 'integer' in props['type']:
                        new_property['type'] = 'integer'
                    elif 'number' in props['type']:
                        new_property['type'] = 'number'
                    elif 'object' in props['type']:
                        # Object arguments should be defined in specification
                        # as node inputs, rather than properties
                        new_property = None
                    else:
                        new_property['type'] = 'text'
                # If no type is specified then text is used
                else:
                    new_property['type'] = 'text'

                if new_property is not None:
                    properties.append(new_property)

            specification['nodes'].append({
                'name': node.name,
                'type': node.type,
                'category': node.category,
                'properties': properties,
                'inputs': strip_io(self.io_mapping[node.type]['inputs']),
                'outputs': strip_io(self.io_mapping[node.type]['outputs'])
            })

        return specification

    def parse_json(self, json_cfg: Dict) -> Any:
        """
        Creates Kenning objects that can be later used for inference
        using JSON config.

        Parameters
        ----------
        json_cfg : Dict
            Kenning JSON dictionary created with `parse_dataflow` method

        Returns
        -------
        Any
            Kenning objects that can be later run with `run_dataflow`
        """
        return NotImplementedError

    def run_dataflow(self, *args, **kwargs):
        """
        Runs Kenning object created with `parse_json` method.
        """
        return NotImplementedError

    def destroy_dataflow(self, *args, **kwargs):
        """
        Destroys Kenning objects allocated with `parse_json` to free
        the resources allocated during initialization.
        """
        return NotImplementedError

    def create_dataflow(self, pipeline: Dict) -> Dict:
        """
        Parses a Kenning JSON into a Pipeline Manager dataflow format
        that can be loaded into the Pipeline Manager editor.

        It is assumed that the passed pipeline is valid.

        Parameters
        ----------
        pipeline : Dict
            Valid Kenning pipeline in JSON.

        Returns
        -------
        Dict :
            Dataflow that is a valid save in Pipeline Manager format.
        """
        raise NotImplementedError

    def parse_dataflow(self, dataflow: Dict) -> Tuple[bool, Union[Dict, str]]:
        """
        Parses a `dataflow` that comes from Pipeline Manager application.
        If any error during parsing occurs it is returned.
        If parsing is successful a kenning pipeline is returned.

        Parameters
        ----------
        dataflow : Dict
            Dataflow that comes from Pipeline Manager application.

        Returns
        -------
        Tuple[bool, Union[Dict, str]] :
            If parsing is successful then (True, pipeline) is returned where
            pipeline is a valid JSON that can be used to run an inference.
            Otherwise (False, error_message) is returned where error_message
            is an error that occured during parsing process.
        """
        raise NotImplementedError

    @staticmethod
    def get_nodes(
        nodes: List[Node] = None,
        io_mapping: Dict = None
    ) -> Tuple[List[Node], Dict]:
        """
        Defines specification for the dataflow type that will be managed
        in Pipeline Manager.

        Parameters
        ----------
        nodes : List[Node], optional
            If None, new list of nodes is created, otherwise all items are
            appended to the provided argument.

        Returns
        -------
        List[Node]:
            List of all available items applicable for the chosen dataflow
            type. It is checked at the runtime whether the item can be loaded
            using specific Kenning configuration, all non available items(for
            example due to lack of needed dependency) are filtered out.

        Dict:
            Mapping used by Pipeline Manager to define the inputs and
            outputs of each node type that will later appear in manager's
            graph.
        """
        raise NotImplementedError
