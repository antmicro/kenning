# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Package that is mocking pkg_resources module.
"""

import importlib_resources


def resource_string(x, y):
    """
    A mock for pkg_resources resource_string function.
    """
    return importlib_resources.files(x).joinpath(y).read_bytes()


def resource_isdir(x, y):
    """
    A mock for pkg_resources resource_isdir function.
    """
    return importlib_resources.files(x).joinpath(y).is_dir()


def resource_listdir(x, y):
    """
    A mock for pkg_resources resource_listdir function.
    """
    return importlib_resources.files(f"{x}.{y}").iterdir()


def resource_stream(x, y):
    """
    A mock for pkg_resources resource_stream function.
    """
    return importlib_resources.files(x).joinpath(y)


def resource_filename(x, y):
    """
    A mock for pkg_resources resource_filename function.
    """
    return importlib_resources.as_file(importlib_resources.files(x) / y)
