# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides additional functions for AutoPyTorch flow.
"""

from typing import (
    Any,
    Dict,
    List,
    Type,
)

import ConfigSpace as CS
from autoPyTorch.utils.common import (
    HyperparameterSearchSpace,
    get_hyperparameter,
)
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

TYPE_TO_CATEGORIES = {
    int: UniformIntegerHyperparameter,
    float: UniformFloatHyperparameter,
    str: CategoricalHyperparameter,
    bool: CategoricalHyperparameter,
    list: UniformIntegerHyperparameter,
}


def _create_forbidden_choices(
    cs: CS.ConfigurationSpace,
    hyperparam: str,
    choices: List[str],
    update_default: bool = False,
) -> CS.ForbiddenInClause:
    """
    Creates ConfigSpace's Forbidden Clause for given hyperparam
    and list of not forbidden choices.

    Parameters
    ----------
    cs : CS.ConfigurationSpace
        Configuration Space containing hyperparameters.
    hyperparam : str
        Name of the hyperparameter.
    choices : List[str]
        List of choices which are not forbidden.
    update_default : bool
        Whether default value of the hyperparameter should be updated.
        Uses first value of the choices.

    Returns
    -------
    CS.ForbiddenInClause
        Created Forbidden Clause
    """
    param = cs.get_hyperparameter(hyperparam)
    if update_default and param.default_value not in choices:
        param.default_value = choices[0]
    return CS.ForbiddenInClause(
        param,
        [v for v in param.choices if v not in choices],
    )


def _add_single_hyperparameter(
    cs: CS.ConfigurationSpace,
    name: str,
    config: Dict[str, Any],
    c_type: Type,
    c_default: Any,
) -> CS.hyperparameters.Hyperparameter:
    """
    Adds hyperparameter based on schema.

    Parameters
    ----------
    cs : CS.ConfigurationSpace
        Configuration Space where hyperparameter will be added.
    name : str
        The name of the parameter.
    config : Dict[str, Any]
        Configuration of the parameter.
    c_type : Type
        Type of the parameter - int, float, str, bool or list.
    c_default : Any
        Default value of the parameter.

    Returns
    -------
    CS.hyperparameters.Hyperparameter
        Newly added hyperparameter
    """
    if c_type in (int, float):
        c_range = config.get("enum", config.get("item_range", None))
    elif c_type is list:
        c_range = config["list_range"]
    elif c_type is str:
        c_range = config["enum"]
    else:  # is bool
        c_range = (True, False)
    if config.get("nullable", False):
        c_range = [*c_range, None]

    param = get_hyperparameter(
        HyperparameterSearchSpace(
            hyperparameter=name,
            value_range=c_range,
            default_value=c_default,
        ),
        CategoricalHyperparameter
        if config.get("enum", None)
        else TYPE_TO_CATEGORIES[c_type],
    )
    cs.add_hyperparameter(param)
    return param
