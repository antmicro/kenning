from kenning.utils.pipeline_manager.dataflow_specification import nodes, io_mapping  # noqa: E501
from kenning.utils import logger
from typing import Union, Dict, Tuple


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


def parse_dataflow(dataflow: Dict) -> Tuple[bool, Union[Dict, str]]:
    """
    Parses a `dataflow` that comes from Pipeline Manager application.
    If any error during parsing occurs it is returned.
    If parsing is successful a kenning scenario is returned.

    Parameters
    ----------
    dataflow : Dict
        Dataflow that comes from Pipeline Manager application.

    Returns
    -------
    Tuple[bool, Union[Dict, str]] :
        If parsing is successful then (True, scenario) is returned where
        scenario is a valid JSON that can be used to run an inference.
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

    scenario = {
        'model_wrapper': [],
        'runtime': [],
        'optimizer': [],
        'dataset': []
    }

    for node in kenning_nodes:
        scenario[node['type']].append(node)

    # Checking cardinality of the nodes
    if len(scenario['dataset']) != 1:
        mes = (
            'Multiple instances of dataset class'
            if len(scenario['dataset']) > 1
            else 'No dataset class instance'
        )
        return return_error(mes)
    if len(scenario['runtime']) != 1:
        mes = (
            'Multiple instances of runtime class'
            if len(scenario['runtime']) > 1
            else 'No runtime class instance'
        )
        return return_error(mes)
    if len(scenario['model_wrapper']) != 1:
        mes = (
            'Multiple instances of model_wrapper class'
            if len(scenario['model_wrapper']) > 1
            else 'No model_wrapper class instance'
        )
        return return_error(mes)

    dataset = scenario['dataset'][0]
    model_wrapper = scenario['model_wrapper'][0]
    runtime = scenario['runtime'][0]
    optimizers = scenario['optimizer']

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
