# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for ConverterRegistry related functions.
"""
from typing import Dict, List

from kenning.core.exceptions import IOSpecificationNotFoundError


def ensure_processed_input(io_spec: Dict[str, List[Dict]]) -> Dict:
    """
    Checks if processed_input is defined.
    If not, sets to to the same val as input.

    Parameters
    ----------
    io_spec : Dict[str, List[Dict]]
        Model io specification.

    Returns
    -------
    Dict
        Io specification with processed_input included.

    Raises
    ------
    IOSpecificationNotFoundError
        It is raised when invalid specification was provided.
    """
    try:
        if "processed_input" not in io_spec:
            io_spec["processed_input"] = io_spec["input"]

        return io_spec
    except KeyError:
        raise IOSpecificationNotFoundError(
            "Neither input nor processed input is defined in io_spec"
        )
