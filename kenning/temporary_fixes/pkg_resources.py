# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Package that is mocking pkg_resources module.
"""

import os
import pkgutil
from pathlib import Path


def resource_string(x, y):
    """
    A mock for pkg_resources resource_string function.
    """
    return pkgutil.get_data(x, y)


def resource_isdir(x, y):
    """
    A mock for pkg_resources resource_isdir function.
    """
    resource_path = resource_filename(x, y)

    return os.path.isdir(resource_path)


def resource_listdir(x, y):
    """
    A mock for pkg_resources resource_listdir function.
    """
    module_path = resource_filename(x, y)

    return os.listdir(module_path)


def resource_stream(x, y):
    """
    A mock for pkg_resources resource_stream function.
    """
    loader = pkgutil.get_loader(x)

    module_path = Path(pkgutil.get_loader(x).get_filename())
    resource_path = os.path.join(module_path.parent, y)

    return loader.get_data(resource_path)


def resource_filename(x, y):
    """
    A mock for pkg_resources resource_filename function.
    """
    module_path = Path(pkgutil.get_loader(x).get_filename())

    return os.path.join(module_path.parent, y)
