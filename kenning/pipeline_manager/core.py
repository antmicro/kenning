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
        Name of the block.
    category : str
        Category of the node (where it appears in the available blocks menu).
    type : str
        Type of the block (for internal usage in pipeline manager).
    cls_name : str
        Full name of class (including module names) that the node represents.
    """
    name: str
    category: str
    type: str
    cls_name: str


class GraphCreator:
    """
    Base class for generating graphs in a particular JSON format.
    """

    def __init__(self):
        self.start_new_graph()
        self._id = -1

    def start_new_graph(self):
        """
        Starts creating new graph.
        """
        self.nodes = {}
        self.reset_graph()

    def gen_id(self) -> str:
        """
        Utility function for unique ID generation.
        """
        self._id += 1
        return str(self._id)

    def reset_graph(self):
        """
        Resets the internal state of graph creator.
        """
        raise NotImplementedError

    def create_node(
            self,
            node: Node,
            parameters: Any
    ) -> str:
        """
        Creates new node in a graph. Graph creator abstracts away the
        implementation specific details of a node representation in a graph,
        which means that the handler only needs to work with the unique ID
        that the graph creator assigns to each node. This ID should be later
        used by a handler to create connections between nodes.

        Parameters
        ----------
        node : Node
            Kenning Node that should be represented in a graph.
        parameters : Any
            Format specific parameters of the graph node.

        Returns
        -------
        str : ID of newly created graph node.
        """
        raise NotImplementedError

    def find_compatible_io(
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
        from_id : str
            ID of starting graph node.
        to_id : str
            ID of ending graph node.

        Returns
        -------
        Tuple[Any, Any] :
            Format specific identification of named input and output. First
            element of tuple is name of output port of a starting node,
            second is identification of input of a node where the connection
            ends.
        """
        raise NotImplementedError

    def create_connection(
            self,
            from_id: str,
            to_id: str
    ):
        """
        Creates connection between two nodes.

        Parameters
        ----------
        from_id : str
            ID of starting graph node.
        to_id : str
            ID of ending graph node.
        """
        raise NotImplementedError

    def flush_graph(self) -> Any:
        """
        Ends and resets graph creation process.

        Returns
        -------
        Any : Finalized graph.
        """
        raise NotImplementedError


class BaseDataflowHandler:
    """
    Base class used for interpretation of graphs coming from Pipeline
    manager. Subclasses are used to define specifics of one of the Kenning
    graph formats (such as Kenningflow or Kenning optimization pipeline).
    Defines conversion to and from Pipeline Manager format, parsing and
    running incoming dataflows with Kenning.
    """

    def __init__(
            self,
            nodes: Dict[str, Node],
            io_mapping: Dict[str, Dict],
            graph_creator: GraphCreator
    ):
        """
        Prepares the dataflow handler, creates graph creators - `pm_graph` for
        creating graph in Pipeline Manager format and `dataflow_graph` for
        creating JSON with a specific Kenning dataflow.

        Parameters
        ----------
        nodes : Dict[str, Node]
            List of available nodes for this dataflow type.
        io_mapping : Dict[str, Dict]
            Mapping used by Pipeline Manager for defining the shape
            of each node type.
        graph_creator : GraphCreator
            Creator used for parsing Pipeline manager dataflows into specific
            JSON format.
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
        Dict: Specification ready to be send to Pipeline Manager.
        """
        specification = {
            'version': '20230619.3',
            'metadata': {
                'twoColumn': True,
            },
            'nodes': []
        }

        def strip_io(io_list: list, direction) -> list:
            """
            Strips every input/output from metadata and leaves only
            `name` and `type` keys.
            """
            return [
                {
                    'name': io['name'],
                    'type': io['type'],
                    'direction': direction
                }
                for io in io_list
            ]

        toremove = set()
        for key, node in self.nodes.items():
            try:
                node_cls = load_class(node.cls_name)
            except (ModuleNotFoundError, ImportError, Exception) as err:
                msg = f'Could not add {node_cls}. Reason:'
                _LOGGER.warn('-' * len(msg))
                _LOGGER.warn(msg)
                _LOGGER.warn(err)
                _LOGGER.warn('-' * len(msg))
                toremove.add(key)
                continue
            parameterschema = node_cls.form_parameterschema()

            properties = []
            for name, props in parameterschema['properties'].items():
                new_property = {'name': name}

                if 'default' in props:
                    new_property['default'] = props['default']

                if 'description' in props:
                    new_property['description'] = props['description']

                def add_default(default_val):
                    if new_property.get('default') is None:
                        new_property['default'] = default_val

                # Case for an input with range defined
                if 'enum' in props:
                    new_property['type'] = 'select'
                    new_property['values'] = list(map(str, props['enum']))
                    add_default(new_property['values'][0])
                # Case for a single value input
                elif 'type' in props:
                    if 'array' in props['type']:
                        new_property['type'] = 'list'
                        if 'items' in props and 'type' in props['items']:
                            dtype = props['items']['type']
                            new_property['dtype'] = dtype
                        add_default([])
                    elif 'boolean' in props['type']:
                        new_property['type'] = 'checkbox'
                        add_default(False)
                    elif 'string' in props['type']:
                        new_property['type'] = 'text'
                        add_default('')
                    elif 'integer' in props['type']:
                        new_property['type'] = 'integer'
                        add_default(0)
                    elif 'number' in props['type']:
                        new_property['type'] = 'number'
                        add_default(0)
                    elif 'object' in props['type']:
                        # Object arguments should be defined in specification
                        # as node inputs, rather than properties
                        new_property = None
                    else:
                        new_property['type'] = 'text'
                        add_default('')
                # If no type is specified then text is used
                else:
                    new_property['type'] = 'text'
                    add_default('')

                if new_property is not None:
                    properties.append(new_property)

            specification['nodes'].append({
                'name': node.name,
                'type': node.type,
                'category': node.category,
                'properties': properties,
                'interfaces': strip_io(
                    self.io_mapping[node.type]['inputs'],
                    'input'
                ) + strip_io(
                    self.io_mapping[node.type]['outputs'],
                    'output'
                )
            })

        for key in toremove:
            del self.nodes[key]
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
            is an error that occurred during parsing process.
        """

        try:
            interface_to_id = {}
            graph = dataflow['graph']
            for dataflow_node in graph['nodes']:
                kenning_node = self.nodes[dataflow_node['type']]
                parameters = dataflow_node['properties']
                parameters = {
                    name: parameter['value']
                    for name, parameter in parameters.items()
                }
                node_id = self.dataflow_graph.create_node(
                    kenning_node,
                    parameters
                )

                for _, input in dataflow_node['inputs'].items():
                    interface_to_id[input['id']] = node_id

                for _, output in dataflow_node['outputs'].items():
                    interface_to_id[output['id']] = node_id

            for conn in graph['connections']:
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
        Parses incoming JSON dataflow into Kenning objects that can be later
        used to run inference (for example flow handler will create
        a Kenningflow instance, while pipeline handler will create Kenning
        objects such as Optimizers, Dataset, Runtime, etc.).

        Parameters
        ----------
        json_cfg : Dict
            Kenning JSON dictionary created with `parse_dataflow` method.

        Returns
        -------
        Any :
            Kenning objects that can be later run with `run_dataflow`.
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

    def create_dataflow(self, pipeline: Dict) -> Dict[str, Union[float, Dict]]:
        """
        Parses a Kenning JSON into a Pipeline Manager dataflow format
        that can be loaded into the Pipeline Manager editor. Should
        utilize Pipeline Manager graph creator to abstract the details
        of graph representation, so the method should only deal with parsing
        the nodes with its parameters and connections between them.

        It is assumed that the passed pipeline is valid.

        For the details of the shape of resulting dictionary, check the
        Pipeline Manager graph representation detailed in
        `PipelineManagerGraphCreator` documentation.

        Parameters
        ----------
        pipeline : Dict
            Valid Kenning pipeline in JSON.

        Returns
        -------
        Dict[str, Union[float, Dict]] :
            JSON representation of a dataflow in Pipeline Manager format.
            Should not be created directly, but rather should be the result of
            `flush_graph` method from graph creator.
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
    """
    Abstraction for graph generation in Pipeline Manager format.

    Graphs in Pipeline Manager are represented in a following JSON dictionary:
    {
        'graph': {
            'nodes' - list of nodes
            'connections' - list of connections
            'inputs' - dictionary of inputs of the main graph
            'outputs' - dictionary of outputs of the main graph
        },
        'graphTemplates: {}
    }

    Each node is represented by a dictionary:
    {
        'type' - type of a node as defined in 'get_nodes' method of
        dataflow handler
        'id' - node's unique ID string
        'title' - title of a node
        'properties' - dictionary of parameters parsed JSON file defining
        specific Kenning module.
        'inputs' - dictionary of input ports of a node,
        'outputs' - dictionary of output ports of a node,
        'position' - dictionary containing two values: 'x' and 'y' that
        define the placement of a node
        'width' - width of a node
        'twoColumn' - boolean value that represents whether the node parameters
        should be splitted into two columns
    }

    Each property defined as a dictionary:
    'name': {
        'value' - value of the property
    }


    Each input and output interface is defined as a dictionary:
    'name': {
        'id' - ID of an interface
    }

    Connection is defined as a dictionary:
    {
        'id' - connection's unique ID,
        'from' - ID of a interface that is a starting point of a connection
        'to' - ID of a interface that is an ending point of a connection
    }
    """

    def __init__(
            self,
            io_mapping: Dict,
            start_x_pos: int = 50,
            start_y_pos: int = 50,
            node_width: int = 300,
            node_x_offset: int = 50
    ):
        """
        Prepares the Graph creator for Pipeline Manager.

        Parameters
        ----------
        io_mapping : Dict[str, Dict]
            IO mapping based on the input nodes specification.
        start_x_pos, start_y_pos : int
            Position of the first graph node.
        node_width : int
            Width of nodes.
        node_x_offset : int
            Spacing between two nodes.
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
        self.interface_map = {}
        self.reset_position()

    def update_position(self):
        """
        Calculates position for a new node based on previous (x,y).
        """
        self.x_pos += self.node_width + self.node_x_offset

    def reset_position(self):
        """
        Returns the position to it's original value.
        """
        self.x_pos = self.start_x_pos

    def _create_interface(
            self,
            io_spec: Dict[str, List],
    ) -> Tuple[str, List]:
        """
        Creates a node interface based on it's IO specification.

        Parameters
        ----------
        io_spec: Dict[str, List]
            IO specification of an input.

        Returns
        -------
        Tuple[str, List]
            Created interface together with its ID.
        """
        interface_id = self.gen_id()
        interface = {
            io_spec['name']: {'id': interface_id}
        }
        return interface_id, interface

    def create_node(self, node, parameters):
        node_id = self.gen_id()
        io_map = self.io_mapping[node.type]

        inputs = {}
        outputs = {}
        for io_spec in io_map['inputs']:
            interface_id, interface = self._create_interface(io_spec)
            inputs |= interface
            self.interface_map[interface_id] = io_spec
        for io_spec in io_map['outputs']:
            interface_id, interface = self._create_interface(io_spec)
            outputs |= interface
            self.interface_map[interface_id] = io_spec

        self.nodes[node_id] = {
            'type': node.name,
            'id': node_id,
            'properties': parameters,
            'inputs': inputs,
            'outputs': outputs,
            'position': {
                'x': self.x_pos,
                'y': self.y_pos,
            },
            'width': self.node_width,
            'twoColumn': False
        }
        self.update_position()
        return node_id

    def find_compatible_io(self, from_id, to_id):
        # TODO: I'm assuming here that there is only one pair of matching
        # input-output interfaces
        from_interface_dict = self.nodes[from_id]['outputs']
        to_interface_dict = self.nodes[to_id]['inputs']

        for from_interface, to_interface in itertools.product(
                from_interface_dict.values(), to_interface_dict.values()):

            from_interface_id = from_interface['id']
            to_interface_id = to_interface['id']
            from_io_spec = self.interface_map[from_interface_id]
            to_io_spec = self.interface_map[to_interface_id]

            if from_io_spec['type'] == to_io_spec['type']:
                return from_interface_id, to_interface_id
        raise RuntimeError("No compatible connections were found")

    def create_connection(self, from_id, to_id):
        from_interface_id, to_interface_id = self.find_compatible_io(
            from_id, to_id
        )
        self.connections.append({
            'id': self.gen_id(),
            'from': from_interface_id,
            'to': to_interface_id
        })

    def flush_graph(self):
        finished_graph = {
            'graph': {
                'id': self.gen_id(),
                'nodes': list(self.nodes.values()),
                'connections': self.connections,
                'inputs': [],
                'outputs': []
            },
            'graphTemplates': []
        }
        self.start_new_graph()
        return finished_graph
