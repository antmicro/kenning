# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing classes related to the inference loop.
"""

import threading
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from kenning.core.dataconverter import DataConverter
from kenning.core.dataset import Dataset
from kenning.core.measurements import Measurements, tagmeasurements
from kenning.core.model import ModelWrapper
from kenning.core.platform import Platform
from kenning.core.protocol import Protocol
from kenning.core.runtime import Runtime
from kenning.protocols.message import MessageType
from kenning.utils.args_manager import ArgumentsHandler
from kenning.utils.logger import KLogger, LoggerProgressBar


class InferenceLoop(ArgumentsHandler, ABC):
    """
    Abstract wrapper of inference loop.
    """

    arguments_structure = {}

    def __init__(
        self,
        dataset: Dataset,
        dataconverter: DataConverter,
        model_wrapper: ModelWrapper,
        platform: Optional[Platform] = None,
        protocol: Optional[Protocol] = None,
        runtime: Optional[Runtime] = None,
    ):
        self._platform = platform
        self._model_wrapper = model_wrapper
        self._dataset = dataset
        self._protocol = protocol
        self._dataconverter: DataConverter = dataconverter
        self._runtime = runtime

    @abstractmethod
    def _prepare(self):
        """
        Prepare inference loop. Called before entering the inference loop.
        """
        ...

    @abstractmethod
    def _cleanup(self):
        """
        Cleanup inference loop. Called after exiting the inference loop.
        """
        ...

    @abstractmethod
    def _run_loop(self, measurements: Measurements):
        """
        Inference loop implementation.

        Parameters
        ----------
        measurements: Measurements
            Measurements collected during inference.
        """
        ...

    def _preprocess(self, X):
        return tagmeasurements("preprocessing")(
            self._dataconverter.to_next_block
        )(X)

    def _postproces(self, Y):
        return tagmeasurements("postprocessing")(
            self._dataconverter.to_previous_block
        )(Y)

    def _pre_loop_hook(self, measurements):
        """
        Hook called between loop prepare and run.
        """
        ...

    def _post_loop_hook(self, measurements):
        """
        Hook called after return from the inference loop
        (skipped on exceptions).
        """
        ...

    def _compute_metrics(self, measurements: Measurements):
        """
        Calculate additional metrics after the inference loop ends.
        """
        ...

    def run(self) -> Measurements:
        """
        Setups, runs and cleanups the inference loop.
        """
        measurements = Measurements()

        KLogger.info("Starting inference loop")

        try:
            self._prepare()
            self._pre_loop_hook(measurements)
            self._run_loop(measurements)
            self._post_loop_hook(measurements)
        except KeyboardInterrupt:
            KLogger.info("Stopping inference...")
        finally:
            self._cleanup()
            self._compute_metrics(measurements)

        return measurements


class SequentialInferenceLoop(InferenceLoop, ABC):
    """
    Generic implementation of a sequential inference loop.
    """

    @abstractmethod
    def _inference_step(self, X) -> tuple[Any, Measurements]:
        """
        Inference step executed on each iteration.
        Accepts pre-processed data and returns predictions.
        """
        ...

    def _run_loop(self, measurements: Measurements):
        with LoggerProgressBar() as logger_progress_bar:
            for X, y in tqdm(
                self._dataset.iter_test(), **logger_progress_bar.kwargs
            ):
                # TODO: should_cancel?
                prepX = self._preprocess(X)
                preds, step_measurements = self._inference_step(prepX)
                if preds is None:
                    break

                measurements += step_measurements
                postPreds = self._postproces(preds)

                measurements += self._dataset._evaluate(postPreds, y)

                self._platform.inference_step_callback()


class RealtimeInferenceLoop(InferenceLoop):
    """
    Generic implementation of a realtime inference loop.
    """

    def _prepare(self):
        ...

    def _cleanup(self):
        self._protocol.disconnect()

    def _run_loop(self, measurements):
        self._platform.renode_pause()

        stop_event = threading.Event()

        feed_thread = threading.Thread(
            target=self._feed_data,
            args=(stop_event, measurements),
            daemon=True,
        )
        collect_thread = threading.Thread(
            target=self._collect_results,
            args=(stop_event, measurements),
            daemon=True,
        )

        try:
            collect_thread.start()
            feed_thread.start()

            while collect_thread.is_alive() and feed_thread.is_alive():
                collect_thread.join(0.2)
                feed_thread.join(0.2)
        except KeyboardInterrupt:
            KLogger.info("Received keyboard interrupt")
        finally:
            stop_event.set()
            KLogger.info("Stopping inference...")
            collect_thread.join()
            feed_thread.join()

    def _feed_data(
        self, stop_event: threading.Event, measurements: Measurements
    ):
        """
        Reads data from the dataset and passes them to the runtime.
        Called in a separate thread.
        """
        ...

    def _collect_results(
        self, stop_event: threading.Event, measurements: Measurements
    ):
        """
        Collects results from the runtime and stores them in measurements.
        Called in a separate thread.
        """

        def handle_result(message_type, payload, flags):
            result_time = self._platform.get_time()
            KLogger.debug("Received output")
            try:
                if self._model_wrapper is not None:
                    output = self._model_wrapper.convert_output_from_bytes(
                        payload
                    )

                result = self._postproces(output)
            except Exception as e:
                KLogger.error(e)
                raise e

            nonlocal measurements
            measurements += {
                "results": [
                    (result_time, list(map(int, np.array(result).flatten())))
                ]
            }

        self._protocol.listen(
            message_type=MessageType.OUTPUT,
            transmission_callback=handle_result,
        )

        stop_event.wait()
        self._protocol.kill_event(MessageType.OUTPUT)
