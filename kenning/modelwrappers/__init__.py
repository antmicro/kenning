# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides model wrappers for various deep learning models.
"""
import os

# needed for python 3.12 compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1"
