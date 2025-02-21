# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides core methods and classes for integrating Kenning
with Pipeline Manager.
"""

import itertools
import json
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

from pipeline_manager.dataflow_builder.dataflow_builder import GraphBuilder
from pipeline_manager.dataflow_builder.dataflow_graph import AttributeType
from pipeline_manager.dataflow_builder.entities import (
    Interface,
    NodeAttributeType,
    get_uuid,
)

from kenning.utils.class_loader import load_class
from kenning.utils.logger import KLogger

SPECIFICATION_VERSION = "20240723.13"


class VisualEditorGraphParserError(Exception):
    """
    Exception occurring when conversion from scenario to graph and vice versa
    fails.
    """

    pass


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


class GraphCreator(ABC):
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
        ...

    @abstractmethod
    def create_node(self, node: Node, parameters: Any) -> str:
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
        str
            ID of newly created graph node.
        """
        ...

    @abstractmethod
    def find_compatible_io(self, from_id: str, to_id: str) -> Tuple[Any, Any]:
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
        Tuple[Any, Any]
            Format specific identification of named input and output. First
            element of tuple is name of output port of a starting node,
            second is identification of input of a node where the connection
            ends.
        """
        ...

    @abstractmethod
    def create_connection(self, from_id: str, to_id: str):
        """
        Creates connection between two nodes.

        Parameters
        ----------
        from_id : str
            ID of starting graph node.
        to_id : str
            ID of ending graph node.
        """
        ...

    @abstractmethod
    def flush_graph(self) -> Any:
        """
        Ends and resets graph creation process.

        Returns
        -------
        Any
            Finalized graph.
        """
        ...


class BaseDataflowHandler(ABC):
    """
    Base class used for interpretation of graphs coming from Pipeline
    manager. Subclasses are used to define specifics of one of the Kenning
    graph formats (such as Kenningflow or Kenning optimization pipeline).
    Defines conversion to and from Pipeline Manager format, parsing and
    running incoming dataflows with Kenning.
    """

    from pipeline_manager.specification_builder import SpecificationBuilder

    def __init__(
        self,
        nodes: Dict[str, Node],
        io_mapping: Dict[str, Dict],
        graph_creator: GraphCreator,
        spec_builder: SpecificationBuilder,
        layout_algorithm: str,
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
            JSON format
        spec_builder : SpecificationBuilder
            SpecificationBuilder object responsible for handling various
            specification operations, like adding node types
        layout_algorithm : str
            Chooses autolayout algorithm to send in metadata specification
        """
        self.nodes = nodes
        self.io_mapping = io_mapping
        self.dataflow_graph = graph_creator
        self.spec_builder = spec_builder
        self.autolayout = layout_algorithm

        # Creates spec and passes as an argument to build a graph
        spec = self.spec_builder.create_and_validate_spec(
            workspacedir=self.spec_builder.assets_dir
        )
        self.pm_graph = PipelineManagerGraphCreator(
            self.io_mapping, specification=spec
        )

    def get_specification(
        self,
        workspace_dir: Path,
        actions: List[Dict[str, str]],
        spec_save_path: Optional[Path] = None,
    ) -> Dict:
        """
        Prepares core-based Kenning classes to be sent to Pipeline Manager.

        For every class in `nodes` it uses its parameterschema to create
        a corresponding dataflow specification.

        Parameters
        ----------
        workspace_dir : Path
            Pipeline Manager's workspace directory
        actions: List[Dict[str, str]]
            Navbar actions available for a given application
        spec_save_path : Optional[Path]
            Path where the generated specification JSON will be saved.

        Returns
        -------
        Dict
            Specification ready to be send to Pipeline Manager.
        """
        self.spec_builder.metadata_add_param("twoColumn", True)
        self.spec_builder.metadata_add_param("layout", self.autolayout)
        self.spec_builder._metadata["navbarItems"] = actions

        def strip_io(io_list: list, direction) -> list:
            """
            Strips every input/output from metadata and leaves only
            `name` and `type` keys.
            """
            return [
                {
                    "name": io["name"],
                    "type": io["type"],
                    "direction": direction,
                }
                for io in io_list
            ]

        # Mapping dtype to type and default
        standard_types = {
            "boolean": ("bool", False),
            "string": ("text", ""),
            "integer": ("integer", 0),
            "number": ("number", 0),
        }

        nodes_to_remove = set()
        for key, node in self.nodes.items():
            try:
                node_cls = load_class(node.cls_name)
            except (ModuleNotFoundError, ImportError, Exception) as err:
                msg = f"Could not add {node_cls}. Reason:"
                KLogger.warning("-" * len(msg))
                KLogger.warning(msg)
                KLogger.warning(err)
                KLogger.warning("-" * len(msg))
                nodes_to_remove.add(key)
                continue
            parameterschema = node_cls.form_parameterschema()

            for name, props in parameterschema["properties"].items():
                new_property = {
                    "name": name,
                    "default": props.get("default"),
                    "description": props.get("description"),
                }

                def add_default(default_val):
                    if new_property.get("default") is None:
                        new_property["default"] = default_val
                    if isinstance(new_property["default"], Path):
                        new_property["default"] = str(new_property["default"])

                # Case for an input with range defined
                if "enum" in props:
                    new_property["type"] = "select"
                    new_property["values"] = list(map(str, props["enum"]))
                    add_default(new_property["values"][0])
                # Case for a single value input
                elif "type" in props:
                    property_type = props["type"][0]

                    if property_type == "object":
                        # Object arguments should be defined in specification
                        # as node inputs, rather than properties
                        new_property = None
                    elif property_type == "array":
                        new_property["type"] = "list"
                        if "items" in props and "type" in props["items"]:
                            new_property["dtype"] = props["items"]["type"][0]
                        else:
                            # Lists cannot have dtype set to None
                            # so string is used by default
                            new_property["dtype"] = "string"
                        add_default([])
                    elif property_type in standard_types:
                        new_property["type"] = standard_types[property_type][0]
                        new_property["dtype"] = property_type
                        add_default(standard_types[property_type][1])
                    else:
                        new_property["type"] = "text"
                        add_default("")
                # If no type is specified then text is used
                else:
                    new_property["type"] = "text"
                    add_default("")

                if new_property is None:
                    continue

                self.spec_builder.add_node_type_property(
                    node.name,
                    propname=new_property["name"],
                    proptype=new_property["type"],
                    default=new_property.get("default"),
                    description=new_property.get("description"),
                    min=new_property.get("min"),
                    max=new_property.get("max"),
                    values=new_property.get("values"),
                    dtype=new_property.get("dtype"),
                )

            stripped_interfaces = strip_io(
                self.io_mapping[node.type]["inputs"], "input"
            ) + strip_io(self.io_mapping[node.type]["outputs"], "output")
            for interface in stripped_interfaces:
                self.spec_builder.add_node_type_interface(
                    node.name,
                    interfacename=interface["name"],
                    interfacetype=interface["type"],
                    direction=interface["direction"],
                )

        specification = self.spec_builder.create_and_validate_spec(
            workspacedir=workspace_dir,
            dump_spec=spec_save_path,
        )

        for key in nodes_to_remove:
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
        Tuple[bool, Union[Dict, str]]
            If parsing is successful then (True, pipeline) is returned where
            pipeline is a valid JSON that can be used to run an inference.
            Otherwise (False, error_message) is returned where error_message
            is an error that occurred during parsing process.
        """
        try:
            interface_to_id = {}
            graph = dataflow["graphs"][0]
            for dataflow_node in graph["nodes"]:
                kenning_node = self.nodes[dataflow_node["name"]]
                parameters = dataflow_node["properties"]
                parameters = {
                    parameter["name"]: parameter["value"]
                    for parameter in parameters
                    if not (
                        isinstance(parameter["value"], str)
                        and parameter["value"] == ""
                    )
                }
                node_id = self.dataflow_graph.create_node(
                    kenning_node, parameters
                )

                for interface in dataflow_node["interfaces"]:
                    interface_to_id[interface["id"]] = node_id

            for conn in graph["connections"]:
                self.dataflow_graph.create_connection(
                    interface_to_id[conn["from"]], interface_to_id[conn["to"]]
                )

            return True, self.dataflow_graph.flush_graph()
        except RuntimeError as e:
            self.dataflow_graph.start_new_graph()
            return False, str(e)

    @abstractmethod
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
        Any
            Kenning objects that can be later run with `run_dataflow`.
        """
        ...

    def run_dataflow(self, *args, **kwargs):
        """
        Runs Kenning object created with `parse_json` method.
        """
        ...

    def destroy_dataflow(self, *args, **kwargs):
        """
        Destroys Kenning objects allocated with `parse_json` to free
        the resources allocated during initialization.
        """
        ...

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
        Dict[str, Union[float, Dict]]
            JSON representation of a dataflow in Pipeline Manager format.
            Should not be created directly, but rather should be the result of
            `flush_graph` method from graph creator.
        """
        ...

    @staticmethod
    @abstractmethod
    def get_nodes(
        spec_builder: SpecificationBuilder,
        nodes: Optional[Dict[str, Node]] = None,
        io_mapping: Optional[Dict[str, Dict]] = None,
    ) -> Tuple[Dict[str, Node], Dict[str, Dict]]:
        """
        Defines specification for the dataflow type that will be managed
        in Pipeline Manager.

        Parameters
        ----------
        spec_builder : SpecificationBuilder
            SpecificationBuilder object that will be used to form the
            specification programmatically
        nodes : Optional[Dict[str, Node]]
            If None, new nodes list is created, otherwise all items are
            added to the provided argument.
        io_mapping : Optional[Dict[str, Dict]]
            If None, new IO map is created, otherwise all items are
            added to the provided argument.

        Returns
        -------
        Tuple[Dict[str, Node], Dict[str, Dict]]
            * Mapping containing all available items applicable for the chosen
              dataflow type. Keys are the names of Kenning modules, values are
              created items. It is checked at the runtime whether the item can
              be loaded using specific Kenning configuration, all non available
              items(for example due to lack of needed dependency) are filtered
              out.
            * Mapping used by Pipeline Manager to define the inputs and
              outputs of each node type that will later appear in manager's
              graph.
        """
        ...


class PipelineManagerGraphCreator:
    """
    Abstraction for graph generation in Pipeline Manager format.

    For the details regarding the dataflow format definition, follow
    documentation of Pipeline Manager
    """

    def __init__(
        self,
        io_mapping: Dict[str, Dict],
        specification: Dict,
        node_width: int = 300,
    ):
        """
        Prepares the Graph creator for Pipeline Manager.

        Parameters
        ----------
        io_mapping : Dict[str, Dict]
            IO mapping based on the input nodes specification
        specification : Dict
            Specification created by SpecificationBuilder
        node_width : int
            Width of nodes
        """
        self.io_mapping = io_mapping
        self.node_width = node_width
        self.specification = specification

        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "spec.json"
            with open(spec_path, "w") as spec_file:
                json.dump(
                    specification,
                    spec_file,
                    indent=4,
                    sort_keys=True,
                    ensure_ascii=False,
                )
            self.graph_builder = GraphBuilder(
                specification=spec_path,
                specification_version=SPECIFICATION_VERSION,
            )
        self.graph = self.graph_builder.create_graph()

        self.inp_interface_map = {}
        self.out_interface_map = {}

    def create_node(self, node, parameters):
        io_map = self.io_mapping[node.type]

        interfaces = []
        for io_spec in io_map["inputs"]:
            interface = Interface(io_spec["name"], "input")
            interfaces.append(interface)
            self.inp_interface_map[interface.id] = io_spec
        for io_spec in io_map["outputs"]:
            interface = Interface(io_spec["name"], "output")
            interfaces.append(interface)
            self.out_interface_map[interface.id] = io_spec

        node_kwargs = {
            "width": self.node_width,
            "properties": [
                {**param, "id": get_uuid()} for param in parameters
            ],
            "interfaces": interfaces,
            "two_column": True,
        }

        node = self.graph.create_node(node.name, **node_kwargs)
        return node.id

    def find_compatible_io(self, from_id, to_id):
        # TODO: I'm assuming here that there is only one pair of matching
        # input-output interfaces

        from_interface_arr = self.graph.get_by_id(
            AttributeType.NODE, from_id
        ).get(NodeAttributeType.INTERFACE)

        to_interface_arr = self.graph.get_by_id(AttributeType.NODE, to_id).get(
            NodeAttributeType.INTERFACE
        )

        for from_interface, to_interface in itertools.product(
            from_interface_arr, to_interface_arr
        ):
            try:
                from_io_spec = self.out_interface_map[from_interface.id]
                to_io_spec = self.inp_interface_map[to_interface.id]
            except KeyError:
                KLogger.debug(
                    f"The connection from {from_interface.id} to "
                    f"{to_interface.id} could not be established."
                )
                continue

            if from_io_spec["type"] == to_io_spec["type"]:
                return from_interface.id, to_interface.id
        raise RuntimeError("No compatible connections were found")

    def create_connection(self, from_id, to_id):
        from_interface_id, to_interface_id = self.find_compatible_io(
            from_id, to_id
        )
        self.graph.create_connection(from_interface_id, to_interface_id)

    def start_new_graph(self):
        self.graph = self.graph_builder.create_graph()

    def flush_graph(self):
        finished_graph = self.graph.to_json()
        raw_graph = self.graph.to_json()
        graph = (
            raw_graph if isinstance(raw_graph, Dict) else json.loads(raw_graph)
        )
        finished_graph = {"version": SPECIFICATION_VERSION, "graphs": [graph]}
        del self.graph_builder.graphs[0]
        self.start_new_graph()
        return finished_graph
