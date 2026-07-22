# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module which provides model visualizations for reports.
"""


import json
from pathlib import Path

import onnx
from onnx import defs
from pipeline_manager.dataflow_builder.dataflow_builder import (
    GraphBuilder,
)
from pipeline_manager.specification_builder import SpecificationBuilder


def _get_layer_information_from_onnx(model: onnx.ModelProto) -> (list, int):
    initializer_map = {init.name: init for init in model.graph.initializer}

    from onnx import numpy_helper

    layer_count = 0
    layers = list()

    max_connections = 1
    output_count = dict()

    for node in model.graph.node:
        schema = defs.get_schema(node.op_type)
        version = schema.since_version
        layer_params = 0
        layer_bytes = 0
        dtypes = set()

        for input_name in node.input:
            if input_name in initializer_map:
                tensor = numpy_helper.to_array(initializer_map[input_name])
                layer_params += tensor.size
                layer_bytes += tensor.nbytes
                dtypes.add(str(tensor.dtype))

        dtype_str = ", ".join(sorted(dtypes)) if dtypes else "-"
        layer_count += 1

        inputs = list(set(node.input) - set(initializer_map))
        max_connections = max(max_connections, len(inputs))

        outputs = node.output
        output_count[outputs[0]] = 1

        for input in inputs:
            if input != "":
                if input in output_count:
                    output_count[input] += 1
                else:
                    output_count[input] = 1

        layers.append(
            {
                "number": layer_count,
                "name": node.name or node.output[0],
                "parameters": layer_params,
                "bytes": layer_bytes,
                "dtype": dtype_str,
                "op_type": node.op_type,
                "op_version": version,
                "input": inputs,
                "output": outputs,
            }
        )

    return (layers, max(max_connections, max(output_count.values())))


def create_visualization_from_onnx(
    model: onnx.ModelProto, savedir: Path
) -> (Path, Path):
    """
    Generate a model visualiation based on a ONNX model using pipeline-manager.

    Parameters
    ----------
    model : onnx.ModelProto
        onnx model used for creatng the visualization.
    savedir : Path
        Path to the directory for saving pipeline-manager files.

    Returns
    -------
    (Path, Path)
        Path to the specification (first) and path to the graph (second).
    """
    # TODO: frontend builder

    layers, max_connections = _get_layer_information_from_onnx(model)

    SPECIFICATION_VERSION = "20260623.14"
    ASSETS_DIRECTORY = Path("./assets")
    WORKSPACE_DIRECTORY = savedir / "pipeline-manager-workspace"

    specification_builder = SpecificationBuilder(
        spec_version=SPECIFICATION_VERSION,
        assets_dir=ASSETS_DIRECTORY,
        check_urls=True,
    )

    MAX_CONNECTION_COUNT = max_connections

    node_types = set()

    specification_builder.add_node_type(name="input")
    specification_builder.add_node_type_category(
        name="input", category="Input"
    )

    specification_builder.add_node_type_interface(
        name="input",
        interfacename="output",
        side="right",
        maxcount=MAX_CONNECTION_COUNT,
    )

    for layer in layers:
        type = layer["op_type"]
        if type in node_types:
            continue
        node_types.add(type)

        specification_builder.add_node_type(name=type)
        specification_builder.add_node_type_interface(
            name=type,
            interfacename=str("input"),
            maxcount=MAX_CONNECTION_COUNT,
        )
        specification_builder.add_node_type_interface(
            name=type,
            interfacename=str("output"),
            maxcount=MAX_CONNECTION_COUNT,
        )

        specification_builder.add_node_type_category(
            name=type, category="Layers"
        )

        specification_builder.add_node_type_property(
            name=type,
            propname="ID",
            proptype="integer",
            default=0,
            hidden=True,
        )
        specification_builder.add_node_type_property(
            name=type,
            propname="operation version",
            proptype="integer",
            default=0,
            hidden=True,
        )
        specification_builder.add_node_type_property(
            name=type, propname="bytes", proptype="integer", default=0
        )
        specification_builder.add_node_type_property(
            name=type, propname="parameters", proptype="integer", default=0
        )
        specification_builder.add_node_type_property(
            name=type, propname="data type", proptype="text", default="-"
        )

    specification_builder.metadata_add_param(
        paramname="readonly", paramvalue=False
    )

    specification_builder.metadata_add_param(
        paramname="connectionStyle", paramvalue="curved"
    )
    specification_builder.metadata_add_param(
        paramname="layout",
        paramvalue="CytoscapeEngine - dagre-network-simplex",
    )

    specification = specification_builder.create_and_validate_spec(
        workspacedir=WORKSPACE_DIRECTORY,
    )
    specification_path = savedir / "specification.json"
    with open(specification_path, "w") as f:
        json.dump(specification, f)

    graph_builder = GraphBuilder(
        specification=specification_builder,
        specification_version=SPECIFICATION_VERSION,
        workspace_directory=WORKSPACE_DIRECTORY,
    )

    graph = graph_builder.create_graph()

    connections = dict()

    def find_property(node, name):
        for property in node.properties:
            if property.name == name:
                return property

    for i, layer in enumerate(layers):
        node = graph.create_node(layer["op_type"])
        node.instance_name = layer["name"]

        node.set_property("ID", layer["number"])

        node.set_property("bytes", layer["bytes"])
        if layer["bytes"] == 0:
            find_property(node, "bytes").hidden = True

        node.set_property("parameters", layer["parameters"])
        if layer["parameters"] == 0:
            find_property(node, "parameters").hidden = True

        node.set_property("data type", layer["dtype"])
        if layer["dtype"] == "-":
            find_property(node, "data type").hidden = True

        node.set_property(
            "operation version",
            layer["op_version"],
        )

        input_interface = node.get_interfaces_by_regex("input")[0]
        output_interface = node.get_interfaces_by_regex("output")[0]

        inputs = layer["input"]
        outputs = layer["output"]

        for output in outputs:
            connections[output] = {"from": output_interface, "to": list()}

        for input in inputs:
            if input == "":
                continue

            if input in connections:
                connections[input]["to"].append(input_interface)
            else:
                input_node = graph.create_node("input")
                input_node_interface = input_node.get_interfaces_by_regex(
                    "output"
                )[0]

                connections[input] = {
                    "from": input_node_interface,
                    "to": [input_interface],
                }

    used_connections = set()
    for connection_type in connections.values():
        from_interface = connection_type["from"]

        for to_interface in connection_type["to"]:
            if (from_interface.id, to_interface.id) not in used_connections:
                graph.create_connection(from_interface, to_interface)
                used_connections.add((from_interface.id, to_interface.id))

    graph_path = savedir / "graph.json"
    graph_builder.save(json_file=graph_path)
    return specification_path, graph_path
