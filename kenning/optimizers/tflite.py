# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for TensorFlow Lite deep learning compiler.
"""

from kenning.optimizers.litert import LiteRTCompiler


class TFLiteCompiler(LiteRTCompiler):
    """
    Ensures backward compatibility for scenarios using TFLiteCompiler.

    For more details check out LiteRTCompiler.
    """

    ...
