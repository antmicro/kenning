# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Script provides method specifications, based on SpecificationBuilder Python
implementation, which are used to generate methods in the
JavaScript equivalent.

The script is not intended to be used outside of its SpecificationBuilder
JS equivalent.
"""


import json
from dis import get_instructions
from inspect import getfullargspec, getmembers, ismethod
from typing import Callable

# if this assertion doesn't pass then 'specBuilder' name might differs from the
# one used in 'specification_builder_pyodide_source.js' in pyodide namespace
# or script is used out of 'specification_builder_pyodide_source.js' context
try:
    test = specBuilder  # noqa: F821
except NameError:
    assert False, "'specBuilder is not defined"

# methods to exclude or include in methods specidication
exclude_for_sure = ["create_and_validate_spec"]
include_for_sure = ["_construct_specification"]


def returns_explicit_none_only(fn: Callable) -> bool:
    """
    Determines if function doesn't have return
    statement or explicitly returns None only.

    Parameters
    ----------
    fn : Callable
        Function to be checked

    Returns
    -------
    bool
        True if doesn't have return statement (in written code)
        or have explicit None return, False otherwise
    """
    instructions = list(get_instructions(fn))
    for i, instr in enumerate(instructions):
        if instr.opname == "RETURN_VALUE":
            prev = instructions[i - 1] if i > 0 else None
            if prev and not (
                prev.opname == "LOAD_CONST" and prev.argval is None
            ):
                return False
    return True


# tuple (name, obj)
methods_names_objs = filter(
    lambda member: member[0] in include_for_sure
    or (
        not (member[0].startswith("_") or member[0] in exclude_for_sure)
        and ismethod(member[1])
    ),
    getmembers(specBuilder),  # noqa: F821
)

# tuple (name, spec, does_return)
methods_names_specs_returns = list(
    map(
        lambda method: (
            method[0],
            getfullargspec(method[1]),
            not returns_explicit_none_only(method[1]),
        ),
        methods_names_objs,
    )
)

methods_specs = [
    {
        "name": method_spec[0],
        "args": method_spec[1].args,
        "defaults": method_spec[1].defaults,
        "doesReturn": method_spec[2],
    }
    for method_spec in methods_names_specs_returns
]
json.dumps(methods_specs)
