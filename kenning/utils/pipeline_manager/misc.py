from typing import Dict, Tuple, Union
from collections import defaultdict as dd

from kenning.utils import logger
from kenning.utils.pipeline_manager.dataflow_specification import io_mapping, nodes  # noqa: E501


def get_specification() -> Dict:
    """
    Prepares core-based Kenning classes to be sent to Pipeline Manager.

    For every class in `nodes` it uses its parameterschema to create
    a corresponding dataflow specification.
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

    for node in nodes:
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
                else:
                    new_property['type'] = 'text'
            # If no type is specified then text is used
            else:
                new_property['type'] = 'text'

            properties.append(new_property)

        specification['nodes'].append({
            'name': node.name,
            'type': node.type,
            'category': node.category,
            'properties': properties,
            'inputs': strip_io(io_mapping[node.type]['inputs']),
            'outputs': strip_io(io_mapping[node.type]['outputs'])
        })

    return specification


def create_dataflow(pipeline: Dict):
    """
    Parses a Kenning pipeline JSON into a Pipeline Manager dataflow format
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
    dataflow_nodes = []
    dataflow = {
        'panning': {
            'x': 0,
            'y': 0,
        },
        'scaling': 1
    }

    def dict_factory() -> Dict:
        """
        Simple wrapper for a default_dict so that values can be assigned
        without creating the entries first.

        Returns
        -------
        Dict :
            Wrapped defaultdict
        """
        return dd(dict_factory)

    def default_to_regular(d: Dict) -> Dict:
        """
        Function that converts dict_factory dictionary into a normal
        python dictionary.

        Parameters
        ----------
        d : Dict
            Dictionary to be converted into a normal python dictionary.

        Returns
        -------
        Dict :
            Converted python dictionary
        """
        if isinstance(d, dd):
            d = {k: default_to_regular(v) for k, v in d.items()}
        return d

    io_mapping_to_id = dict_factory()
    id = 0

    def get_id() -> int:
        """
        Generator for new unique id values.
        Every call returns a value increased by 1.

        Returns
        -------
        int :
            New unique id
        """
        nonlocal id
        id += 1
        return id

    node_x_offset = 50
    node_width = 300
    x_pos = 0
    y_pos = 50

    def get_new_x_position() -> int:
        """
        Generator for new x node positions.
        Every call returns a value increased by `node_width` and
        `node_x_offset`.

        Returns
        -------
        int
            Newly generated x position.
        """
        nonlocal x_pos
        x_pos += (node_width + node_x_offset)
        return x_pos

    def add_block(kenning_block: dict, kenning_block_name: str):
        """
        Adds block entry to the dataflow definition based on the
        `kenning_block` and `kenning_block_name` arguments.

        Additionaly modifies `io_mapping_to_id` dictionary that saves ids of
        inputs and outputs of every block that is later used to create
        connections between the blocks.

        Parameters
        ----------
        kenning_block : dict
            Dictionary of a block that comes from the definition
            of the pipeline.
        kenning_block_name : str
            Name of the block from the pipeline. Valid values are based on the
            `io_mapping` dictionary.
        """
        _, cls_name = kenning_block['type'].rsplit('.', 1)
        kenning_node = [node for node in nodes if node.name == cls_name][0]

        new_node = {
            'name': kenning_node.name,
            'type': kenning_node.name,
            'id': get_id(),
            'options': [
                [
                    key,
                    value
                ]
                for key, value in kenning_block['parameters'].items()
            ],
            'width': node_width,
            'position': {
                'x': get_new_x_position(),
                'y': y_pos
            },
            'state': {},
        }

        interfaces = []
        for io_name in ['inputs', 'outputs']:
            for io_object in io_mapping[kenning_block_name][io_name]:
                id = get_id()
                interfaces.append([
                    io_object['name'],
                    {
                        'value': None,
                        'isInput': True,
                        'type': io_object['type'],
                        'id': id
                    }
                ])

                io_to_ids = io_mapping_to_id[kenning_block_name][io_name]
                if kenning_block_name == 'optimizer':
                    if io_to_ids[io_object['type']]:
                        io_to_ids[io_object['type']].append(id)
                    else:
                        io_to_ids[io_object['type']] = [id]
                else:
                    io_to_ids[io_object['type']] = id

        new_node['interfaces'] = interfaces
        dataflow_nodes.append(new_node)

    # Add dataset, model_wrapper blocks
    for name in ['dataset', 'model_wrapper']:
        add_block(pipeline[name], name)

    # Add optimizer blocks
    for optimizer in pipeline['optimizers']:
        add_block(optimizer, 'optimizer')

    # Add runtime block
    add_block(pipeline['runtime'], 'runtime')

    if 'runtime_protocol' in pipeline:
        add_block(pipeline['runtime_protocol'], 'runtime_protocol')

    io_mapping_to_id = default_to_regular(io_mapping_to_id)
    # This part of code is strongly related to the definition of io_mapping
    # It should be double-checked if io_mapping changes. For now this is
    # manually set, but can be altered to use io_mapping later on
    connections = []
    connections.append({
        'id': get_id(),
        'from': io_mapping_to_id['dataset']['outputs']['dataset'],
        'to': io_mapping_to_id['model_wrapper']['inputs']['dataset'],
    })
    connections.append({
        'id': get_id(),
        'from': io_mapping_to_id['model_wrapper']['outputs']['model_wrapper'],
        'to': io_mapping_to_id['runtime']['inputs']['model_wrapper'],
    })
    # Connecting the first optimizer with model_wrapper
    connections.append({
        'id': get_id(),
        'from': io_mapping_to_id['model_wrapper']['outputs']['model'],
        'to': io_mapping_to_id['optimizer']['inputs']['model'][0],
    })

    # Connecting optimizers
    for i in range(len(pipeline['optimizers']) - 1):
        connections.append({
            'id': get_id(),
            'from': io_mapping_to_id['optimizer']['outputs']['model'][i],
            'to': io_mapping_to_id['optimizer']['inputs']['model'][i + 1],
        })

    # Connecting the last optimizer with runtime
    connections.append({
        'id': get_id(),
        'from': io_mapping_to_id['optimizer']['outputs']['model'][-1],
        'to': io_mapping_to_id['runtime']['inputs']['model'],
    })

    if 'runtime_protocol' in io_mapping_to_id:
        connections.append({
            'id': get_id(),
            'from': io_mapping_to_id['runtime_protocol']['outputs']['runtime_protocol'],  # noqa: E501
            'to': io_mapping_to_id['runtime']['inputs']['runtime_protocol'],
        })

    dataflow['connections'] = connections
    dataflow['nodes'] = dataflow_nodes
    return dataflow


def parse_dataflow(dataflow: Dict) -> Tuple[bool, Union[Dict, str]]:
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
    log = logger.get_logger()

    def return_error(msg: str) -> Tuple[bool, str]:
        """
        Logs `msg` and returns a Tuple[bool, msg]

        Parameters
        ----------
        msg : str
            Message that is logged and that is returned as a feedback message.

        Returns
        -------
        Tuple[bool, str] :
            Tuple that means that parsing was unsuccessful.
        """
        log.error(msg)
        return False, msg

    def strip_node(node: Dict) -> Dict:
        """
        This method picks only `type` and `parameters` keys for the node.

        Returns
        -------
        Dict :
            Dict that contains `module` and `parameters`
            values of the input `node`.
        """
        return {
            'type': node['module'],
            'parameters': node['parameters']
        }

    def get_connected_node(
            socket_id: str,
            edges: Dict,
            nodes: Dict) -> Dict:
        """
        Get node connected to a given socket_id that is the nodes.

        Parameters
        ----------
        socket_id: str
            Socket for which node is searched
        edges: Dict
            Edges to look for the connection
        nodes: Dict
            Nodes to look for the connected node

        Returns
        -------
        Dict :
            Node that is in `nodes` dictionary, is connected to the
            `socket_id` with an edge that is in `edges` dictionary.
            None otherwise
        """
        connection = [
            edge for edge in edges
            if socket_id == edge['from'] or socket_id == edge['to']
        ]

        if not connection:
            return None

        connection = connection[0]
        corresponding_socket_id = (
            connection['to']
            if connection['from'] == socket_id else
            connection['from']
        )

        for node in nodes:
            for inp in node['inputs']:
                if corresponding_socket_id == inp['id']:
                    return node

            for out in node['outputs']:
                if corresponding_socket_id == out['id']:
                    return node

        # The node connected wasn't int the `nodes` argument
        return None

    dataflow_nodes = dataflow['nodes']
    dataflow_edges = dataflow['connections']

    # Creating a list of every node with its kenning path and parameters
    kenning_nodes = []
    for dn in dataflow_nodes:
        kenning_node = [node for node in nodes if node.name == dn['name']][0]
        kenning_parameters = dn['options']

        parameters = {name: value for name, value in kenning_parameters}
        kenning_nodes.append({
            'module': f'{kenning_node.cls.__module__}.{kenning_node.name}',
            'type': kenning_node.type,
            'parameters': parameters,
            'inputs': [
                interface[1]
                for interface in dn['interfaces']
                if interface[1]['isInput']
            ],
            'outputs': [
                interface[1]
                for interface in dn['interfaces']
                if not interface[1]['isInput']
            ]
        })

    pipeline = {
        'model_wrapper': [],
        'runtime': [],
        'optimizer': [],
        'dataset': []
    }

    for node in kenning_nodes:
        pipeline[node['type']].append(node)

    # Checking cardinality of the nodes
    if len(pipeline['dataset']) != 1:
        mes = (
            'Multiple instances of dataset class'
            if len(pipeline['dataset']) > 1
            else 'No dataset class instance'
        )
        return return_error(mes)
    if len(pipeline['runtime']) != 1:
        mes = (
            'Multiple instances of runtime class'
            if len(pipeline['runtime']) > 1
            else 'No runtime class instance'
        )
        return return_error(mes)
    if len(pipeline['model_wrapper']) != 1:
        mes = (
            'Multiple instances of model_wrapper class'
            if len(pipeline['model_wrapper']) > 1
            else 'No model_wrapper class instance'
        )
        return return_error(mes)

    dataset = pipeline['dataset'][0]
    model_wrapper = pipeline['model_wrapper'][0]
    runtime = pipeline['runtime'][0]
    optimizers = pipeline['optimizer']

    # Checking required connections between found nodes
    for current_node in kenning_nodes:
        inputs = current_node['inputs']
        outputs = current_node['outputs']
        node_type = current_node['type']

        for mapping, io in zip(
            [io_mapping[node_type]['inputs'], io_mapping[node_type]['outputs']],  # noqa: E501
            [inputs, outputs]
        ):
            for connection in mapping:
                if not connection['required']:
                    continue

                corresponding_socket = [
                    inp for inp in io
                    if inp['type'] == connection['type']
                ][0]

                if get_connected_node(
                    corresponding_socket['id'],
                    dataflow_edges,
                    kenning_nodes
                ) is None:
                    return return_error(f'There is no required connection for {node_type} class')  # noqa: E501

    # finding order of optimizers
    previous_block = model_wrapper
    next_block = None
    ordered_optimizers = []
    while True:
        out_socket_id = [
            output['id']
            for output in previous_block['outputs']
            if output['type'] == 'model'
        ][0]

        next_block = get_connected_node(out_socket_id, dataflow_edges, optimizers)  # noqa: E501

        if next_block is None:
            break

        if next_block in ordered_optimizers:
            return return_error('Cycle in the optimizer connections')

        ordered_optimizers.append(next_block)
        previous_block = next_block
        next_block = None

    if len(ordered_optimizers) != len(optimizers):
        return return_error('Cycle in the optimizer connections')

    final_scenario = {
        'model_wrapper': strip_node(model_wrapper),
        'optimizers': [strip_node(optimizer) for optimizer in ordered_optimizers],  # noqa: E501
        'runtime': strip_node(runtime),
        'dataset': strip_node(dataset)
    }

    return True, final_scenario
