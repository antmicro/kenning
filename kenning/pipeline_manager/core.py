# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
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


class GraphCreator:
    """
    Base class for generating graphs in a particular JSON format.
    """
    def __init__(self):
        self.start_new_graph()
        self._id = -1

    def start_new_graph(self):
        """
        Starts creating new graph
        """
        self.nodes = {}
        self.reset_graph()

    def gen_id(self) -> str:
        """
        Utility function for unique ID generation
        """
        self._id += 1
        return str(self._id)

    def reset_graph(self):
        """
        Resets the internal state of graph creator
        """
        raise NotImplementedError

    def create_node(
            self,
            node: Node,
            parameters: Any
    ) -> str:
        """
        Creates new node in a graph

        Parameters
        ----------
        node : Node
            Kenning Node that should be represented in a graph
        parameters : Any
            Format specific parameters of the graph node

        Returns
        -------
        str :
            ID of newly created graph node
        """
        raise NotImplementedError

    def find_compatible_IO(
            self,
            from_id: str,
            to_id: str
    ) -> Tuple[Any, Any]:
        """
        Some graph formats have inputs/outputs of nodes that have a name, or
        are otherwise identifiable. That means that whenever a node has a
        connection, there needs to be a check to what IO the connection is
        associated with. This method finds the pair of matching IO names
        that are utilized by a particular connection.

        Parameters
        ----------
        from_id, to_id :
            IDs of connected graph nodes

        Returns
        -------
        Tuple[Any, Any] :
            Format specific identification of named input and output
        """
        raise NotImplementedError

    def create_connection(
            self,
            from_id: str,
            to_id: str
    ):
        """
        Creates connection between two nodes

        Parameters
        ----------
        from_id, to_id :
            IDs of graph nodes to be connected
        """
        raise NotImplementedError

    def flush_graph(self) -> Any:
        """
        Ends and resets graph creation process

        Returns
        -------
        Any :
            Finalized graph
        """
        raise NotImplementedError


class BaseDataflowHandler:
    def __init__(
            self,
            nodes: List[Node],
            io_mapping: Dict[str, Dict],
            graph_creator: GraphCreator
    ):
        """
        Base class for handling different types of kenning specification,
        converting to and from Pipeline Manager formats, and running them
        in Kenning.

        Parameters
        ----------
        nodes : List[Node]
            List of available nodes for this dataflow type
        io_mapping : Dict[str, Dict]
            Mapping used by Pipeline Manager for defining the shape
            of each node type
        graph_creator : GraphCreator
            Creator used for parsing Pipeline manager dataflows into specific
            JSON format
        """
        self.nodes = nodes
        self.io_mapping = io_mapping
        self.pm_graph = PipelineManagerGraphCreator(io_mapping)
        self.dataflow_graph = graph_creator

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

        for node in self.nodes.values():
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

        try:
            interface_to_id = {}
            for dn in dataflow['nodes']:
                kenning_node = self.nodes[dn['name']]
                parameters = dn['options']
                parameters = {
                    arg: value for arg, value in parameters
                }
                node_id = self.dataflow_graph.create_node(
                    kenning_node,
                    parameters
                )
                for _, interf in dn['interfaces']:
                    interface_to_id[interf['id']] = node_id

            for conn in dataflow['connections']:
                self.dataflow_graph.create_connection(
                    interface_to_id[conn['from']],
                    interface_to_id[conn['to']]
                )

            return True, self.dataflow_graph.flush_graph()
        except RuntimeError as e:
            self.dataflow_graph.start_new_graph()
            return False, str(e)

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
        Any :
            Kenning objects that can be later run with `run_dataflow`
        """
        raise NotImplementedError

    def run_dataflow(self, *args, **kwargs):
        """
        Runs Kenning object created with `parse_json` method.
        """
        raise NotImplementedError

    def destroy_dataflow(self, *args, **kwargs):
        """
        Destroys Kenning objects allocated with `parse_json` to free
        the resources allocated during initialization.
        """
        raise NotImplementedError

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

    @staticmethod
    def get_nodes(
            nodes: Dict[str, Node] = None,
            io_mapping: Dict[str, Dict] = None
    ) -> Tuple[Dict[str, Node], Dict[str, Dict]]:
        """
        Defines specification for the dataflow type that will be managed
        in Pipeline Manager.

        Parameters
        ----------
        nodes : Dict[str, Node], optional
            If None, new nodes list is created, otherwise all items are
            added to the provided argument.

        io_mapping : Dict[str, Dict], optional
            If None, new IO map is created, otherwise all items are
            added to the provided argument.

        Returns
        -------
        Dict[str, Node]:
            Mapping containing all available items applicable for the chosen
            dataflow type. Keys are the names of Kenning modules, values are
            created items. It is checked at the runtime whether the item can
            be loaded using specific Kenning configuration, all non available
            items(for example due to lack of needed dependency) are filtered
            out.

        Dict[str, Dict]:
            Mapping used by Pipeline Manager to define the inputs and
            outputs of each node type that will later appear in manager's
            graph.
        """
        raise NotImplementedError


class PipelineManagerGraphCreator(GraphCreator):
    def __init__(
            self,
            io_mapping: Dict,
            start_x_pos: int = 50,
            start_y_pos: int = 50,
            node_width: int = 300,
            node_x_offset: int = 50
    ):
        """
        Creates graph in the Pipeline Manager dataflow format.

        Parameters
        ----------
        io_mapping : Dict[str, Dict]
            IO mapping based on the input nodes specification
        start_x_pos, start_y_pos : int
            Position of the first graph node
        node_width : int
            Width of nodes
        node_x_offset : int
            Spacing between two nodes
        """
        self.start_x_pos = start_x_pos
        self.x_pos = start_x_pos
        self.y_pos = start_y_pos
        self.node_width = node_width
        self.node_x_offset = node_x_offset
        self.io_mapping = io_mapping
        super().__init__()

    def reset_graph(self):
        self.connections = []
        self.interf_map = {}
        self.reset_position()

    def update_position(self):
        """
        Calculates position for a new node based on previous (x,y)
        """
        self.x_pos += self.node_width + self.node_x_offset

    def reset_position(self):
        """
        Returns the position to it's original value
        """
        self.x_pos = self.start_x_pos

    def _create_interface(
            self,
            io_spec: Dict[str, List],
            is_input: bool
    ) -> Tuple[str, List]:
        """
        Creates a node interface based on it's IO specification

        Parameters
        ----------
        io_spec: Dict[str, List]
            IO specification of an input
        is_input: bool
            True if interface is an input of the node

        Returns
        -------
        Tuple[str, List]
            Created interface together with its ID
        """
        interf_id = self.gen_id()
        interface = [io_spec['name'], {
            'id': interf_id,
            'value': None,
            'isInput': is_input,
            'type': io_spec['type']
        }]
        return interf_id, interface

    def create_node(self, node, parameters):
        node_id = self.gen_id()
        io_map = self.io_mapping[node.type]

        interfaces = []
        for io_spec in io_map['inputs']:
            interf_id, interface = self._create_interface(io_spec, True)
            interfaces.append(interface)
            self.interf_map[interf_id] = io_spec
        for io_spec in io_map['outputs']:
            interf_id, interface = self._create_interface(io_spec, False)
            interfaces.append(interface)
            self.interf_map[interf_id] = io_spec

        self.nodes[node_id] = {
            'type': node.name,
            'id': node_id,
            'name': node.name,
            'options': parameters,
            'state': {},
            'interfaces': interfaces,
            'position': {
                'x': self.x_pos,
                'y': self.y_pos,
            },
            'width': self.node_width,
            'twoColumn': False,
            'customClasses': ""
        }
        self.update_position()
        return node_id

    def find_compatible_IO(self, from_id, to_id):
        # TODO: I'm assuming here that there is only one pair of matching
        # input-output interfaces
        from_interf_arr = self.nodes[from_id]['interfaces']
        to_interf_arr = self.nodes[to_id]['interfaces']
        for (_, from_interf), (_, to_interf) in itertools.product(
                from_interf_arr, to_interf_arr):
            from_interf_id, to_interf_id = from_interf['id'], to_interf['id']
            from_io_spec = self.interf_map[from_interf_id]
            to_io_spec = self.interf_map[to_interf_id]
            if not from_interf['isInput'] \
                    and to_interf['isInput'] \
                    and from_io_spec['type'] == to_io_spec['type']:
                return from_interf_id, to_interf_id
        raise RuntimeError("No compatible connections were found")

    def create_connection(self, from_id, to_id):
        from_interf_id, to_interf_id = self.find_compatible_IO(
            from_id, to_id
        )
        self.connections.append({
            'id': self.gen_id(),
            'from': from_interf_id,
            'to': to_interf_id
        })

    def flush_graph(self):
        finished_graph = {
            'panning': {
                'x': 0,
                'y': 0
            },
            'scaling': 1,
            'nodes': list(self.nodes.values()),
            'connections': self.connections
        }
        self.start_new_graph()
        return finished_graph
