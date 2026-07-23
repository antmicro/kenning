# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime implementation for TFLite models.
"""

from kenning.runtimes.litert import LiteRTRuntime


class TFLiteRuntime(LiteRTRuntime):
    """
    Ensures backward compatibility for scenarios using TFLiteRuntime.

    For more details check out LiteRTRuntime.
    """

    ...
