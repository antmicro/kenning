# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for IREE related functions.
"""

from typing import List, Optional


def parse_args(args: Optional[List[str]]) -> List[str]:
    """
    Adds "--" prefix to arguments if necessary.

    Parameters
    ----------
    args : Optional[List[str]]
        The list of args to prefix.

    Returns
    -------
    List[str]
        The list of args prefixed with "--" exactly once.
    """
    if args is None:
        return []

    return [f"--{option.lstrip('-')}" for option in args]
