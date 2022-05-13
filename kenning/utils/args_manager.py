"""
Module for preparing and serializing class arguments.
"""

import json
from pathlib import Path


def serialize(obj: object) -> str:
    """
    Serializes the given object into a JSON format. 
    
    It serialize all variables mentioned in the dictionary 
    returned by a `form_parameterschema` function.

    Note that object has to implement `form_parameterschema` 
    to be serializable.

    Parameters
    ----------
    obj : object
        Object to serialize

    Returns
    -------
    str: Serialized object in a valid JSON format.
    """
    if hasattr(obj, 'form_parameterschema'):
        to_serialize = obj.form_parameterschema()['properties'].keys()
    else:
        return '{}'

    serialized_dict = {}

    for name, value in vars(obj).items():
        if name in to_serialize:
            serialized_dict[name] = value

    class ArgumentEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, Path):
                return str(obj)
            return json.JSONEncoder.default(self, obj)

    return json.dumps(serialized_dict, cls=ArgumentEncoder)
