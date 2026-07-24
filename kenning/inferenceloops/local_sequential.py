# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing a local implementation of the sequential inference loop.
"""

from kenning.core.inferenceloop import SequentialInferenceLoop
from kenning.core.measurements import Measurements


class LocalSequentialInferenceLoop(SequentialInferenceLoop):
    """
    Local implementation of the sequential inference loop.
    """

    def _prepare(self):
        if self._runtime is not None:
            self._runtime.inference_session_start()
            self._runtime.prepare_local()

    def _cleanup(self):
        if self._runtime is not None:
            self._runtime.inference_session_end()

    def _inference_step(self, X):
        succeed = self._runtime.load_input(X)
        if not succeed:
            return None, Measurements()

        self._runtime._run()
        preds = self._runtime.extract_output()

        return preds, Measurements()
