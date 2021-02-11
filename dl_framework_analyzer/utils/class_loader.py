import importlib
from typing import ClassVar


def load_class(modulepath: str) -> ClassVar:
    """
    Loads class given in the module path.

    Parameters
    ----------
    modulepath : str
        Module-like path to the class
    """
    module_name, cls_name = modulepath.rsplit('.', 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    return cls
