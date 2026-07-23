# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module for TVM related functions.
"""


class RepeatGeneratorWrapper:
    """
    Class wprapping a generator function so that it can be used multiple times.
    """

    def __init__(self, generator_func: callable):
        """
        Class wprapping a generator function so that it can be used multiple
        times.

        Each time __iter__ is called on this class, a fresh inner generator is
        created.

        Parameters
        ----------
        generator_func : callable
            The function returning a generator to be wrapped.
        """
        self.generator_func = generator_func
        self.generator = generator_func()

    def __iter__(self):
        self.generator = self.generator_func()
        return self

    def __next__(self):
        return next(self.generator)
