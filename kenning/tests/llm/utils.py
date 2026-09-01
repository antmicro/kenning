# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper code for LLM tests.
"""

from typing import List, Type

from kenning.core.optimizer import Optimizer
from kenning.modelwrappers.llm.llm import LLM


def models_list() -> List[Type[LLM]]:
    from kenning.utils.class_loader import get_all_subclasses

    models = get_all_subclasses(
        "kenning.modelwrappers.llm",
        LLM,
        raise_exception=True,
    )

    return models


def optimizer_list() -> List[Type[Optimizer]]:
    import contextlib

    from kenning.optimizers.awq import AWQOptimizer
    from kenning.optimizers.gptq import GPTQOptimizer

    optimizers = [
        AWQOptimizer,
        GPTQOptimizer,
    ]

    with contextlib.suppress(ImportError):
        from kenning.optimizers.gptq_sparsegpt import GPTQSparseGPTOptimizer

        optimizers.append(GPTQSparseGPTOptimizer)

    return optimizers
