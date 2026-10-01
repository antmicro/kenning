# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module with helper functions for JS SpecificationBuilder to adapt
stringify argument struct to Python.
"""

import json
from typing import Dict, List


def prepare_args(args_str: str, max_args_number: int) -> tuple[List, Dict]:
    """
    Prepares mandatory and optional arguments.

    Prepared arguments are ready to be passed
    into python's method call in a following manure
    mandatory: [req_arg1, req_arg2, ...]
    optional: {opt_arg1: value1, opt_arg2: value2, ...}

    Parameters
    ----------
    args_str: str
        String representing the argument structure
        and derived from JS SpecificationBuilder.
    max_args_number: int
        Integer indicating maximum argument number in 'args_str' argument.

    Returns
    -------
    tuple[List, Dict]
        Pair of mandatory arguments List and optional arguments Dict
    """
    mandatory_args = []
    optioal_args = {}

    args = json.loads(args_str, strict=False)

    if len(args) == max_args_number:
        mandatory_args = args[: max_args_number - 1]
        optioal_args = args[max_args_number - 1]
    else:
        mandatory_args = args

    return mandatory_args, optioal_args
