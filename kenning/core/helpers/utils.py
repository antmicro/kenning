# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Collection of methods for educing redundancies.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_model_size(
    model_path: Path, exception: Optional[Exception] = None
) -> float:
    """
    Returns the model size.

    By default, the size of file with model is returned.

    Parameters
    ----------
    model_path: Path
        Path to the model.
    exception: Optional[Exception]
        Exception to be raised in case of model_path
        doesn't exist.

    Returns
    -------
    float
        The size of the optimized model in KB.

    Raises
    ------
    Exception
        If model size cannot be retrieved.
    """
    if not model_path.exists():
        if exception:
            raise exception
        else:
            raise Exception(f"Model path does not exist: {model_path}")
    return model_path.stat().st_size / 1024


def is_list_of_dicts(value: Any) -> bool:
    """
    Check if provided value if a list of dictionaries.

    Parameters
    ----------
    value : Any
        Value to be checked.

    Returns
    -------
    bool
        Whether the provided value if a list of dictionaries.
    """
    return (
        isinstance(value, List)
        and len(value) > 0
        and isinstance(value[0], Dict)
    )
