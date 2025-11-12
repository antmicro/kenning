# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing a remote implementation of the sequential inference loop.
"""

from pathlib import Path
from typing import Optional

from kenning.core.dataconverter import DataConverter
from kenning.core.dataset import Dataset
from kenning.core.exceptions import ModelTooLargeError
from kenning.core.inferenceloop import SequentialInferenceLoop
from kenning.core.model import ModelWrapper
from kenning.core.platform import Platform
from kenning.core.protocol import Protocol, check_request
from kenning.core.runtime import Runtime
from kenning.utils.logger import KLogger


class RemoteSequentialInferenceLoop(SequentialInferenceLoop):
    """
    Remote implementation of the sequential inference loop.
    """

    arguments_structure = {
        "model_path": {
            "argparse_name": "--model-path",
            "description": "Model path for remote sequential inference loop",
            "type": str,
            "required": True,
        }
    }

    def __init__(
        self,
        dataset: Dataset,
        dataconverter: DataConverter,
        model_wrapper: ModelWrapper,
        platform: Optional[Platform] = None,
        protocol: Optional[Protocol] = None,
        runtime: Optional[Runtime] = None,
        model_path: str = "",
    ):
        super().__init__(
            dataset, dataconverter, model_wrapper, platform, protocol, runtime
        )

        self._model_path = Path(model_path)

    def _prepare(self):
        compiled_model_path = None
        if self._runtime is not None:
            spec_path = self._runtime.get_io_spec_path(self._model_path)
            if not spec_path.exists():
                KLogger.error("No Input/Output specification found")
                raise FileNotFoundError("IO specification not found")
            if (ram_kb := getattr(self._platform, "ram_size_kb", None)) and (
                (model_kb := self._model_path.stat().st_size // 1024) > ram_kb
            ):
                KLogger.error(
                    f"Model ({model_kb}KB) does not fit "
                    f"into board's RAM ({ram_kb}KB)"
                )
                raise ModelTooLargeError(
                    f"Model too large ({model_kb}KB > {ram_kb}KB)"
                )

            check_request(
                self._protocol.upload_io_specification(spec_path),
                "upload io spec",
            )

            compiled_model_path = (
                self._runtime.preprocess_model_to_upload(self._model_path)
                if self._model_path is not None
                else None
            )

        check_request(
            self._protocol.upload_model(compiled_model_path), "upload model"
        )

    def _cleanup(self):
        self._protocol.disconnect()

    def _post_loop_hook(self, measurements):
        measurements += self._protocol.download_statistics(final=True)

    def _inference_step(self, X):
        if self._model_wrapper is not None:
            prepX = self._model_wrapper.convert_input_to_bytes(X)
        else:
            prepX = X
        check_request(self._protocol.upload_input(prepX), "send input")
        check_request(
            self._protocol.request_processing(self._platform.get_time),
            "inference",
        )
        _, preds = check_request(
            self._protocol.download_output(), "receive output"
        )
        KLogger.debug("Received output")
        if self._model_wrapper is not None:
            preds = self._model_wrapper.convert_output_from_bytes(preds)

        measurements = self._protocol.download_statistics(final=False)

        return preds, measurements
