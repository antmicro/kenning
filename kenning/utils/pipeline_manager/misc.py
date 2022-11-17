from kenning.utils.pipeline_manager.dataflow_specification import nodes, io_mapping  # noqa: E501


def get_specification():
    """
    Prepares core-based Kenning classes to be sent to Pipeline Manager.

    For every class in `nodes` it uses its parameterschema to create
    a corresponding dataflow specification.
    """
    specification = {
        'metadata': {},
        'nodes': []
    }

    def strip_io(io_list: list):
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
