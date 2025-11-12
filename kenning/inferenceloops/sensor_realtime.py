# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing a sensor implementation of the real-time inference loop.
"""

import threading
from functools import reduce
from typing import Optional

import numpy as np

from kenning.core.dataconverter import DataConverter
from kenning.core.dataset import Dataset
from kenning.core.inferenceloop import RealtimeInferenceLoop
from kenning.core.measurements import Measurements
from kenning.core.model import ModelWrapper
from kenning.core.platform import Platform
from kenning.core.protocol import Protocol
from kenning.core.runtime import Runtime
from kenning.core.sensor import Sensor
from kenning.platforms.simulatable_platform import SimulatablePlatform


class SensorRealtimeInferenceLoop(RealtimeInferenceLoop):
    """
    Sensor implementation of the real-time inference loop.
    """

    _platform: SimulatablePlatform

    def __init__(
        self,
        dataset: Dataset,
        dataconverter: DataConverter,
        model_wrapper: ModelWrapper,
        platform: Optional[Platform] = None,
        protocol: Optional[Protocol] = None,
        runtime: Optional[Runtime] = None,
    ):
        super().__init__(
            dataset, dataconverter, model_wrapper, platform, protocol, runtime
        )
        self._sensors: list[Sensor] = []
        self._frequency = self._platform.sensors_frequency

    def _prepare(self):
        for sensor_path in self._platform.sensors:
            sensor = reduce(
                lambda x, y: getattr(x, y),
                sensor_path.split("."),
                self._platform.machine.sysbus,
            )
            self._sensors.append(Sensor.create_from_peripheral(sensor))

    def _feed_data(
        self, stop_event: threading.Event, measurements: Measurements
    ):
        for sample in list(self._dataset.iter_test()):
            X, y = sample
            X = list(map(float, X[0][0][-1]))

            idx = 0

            for sensor in self._sensors:
                sensor.feed(X[idx : idx + sensor.size()])
                idx += sensor.size()

            sample_time = self._platform.get_time()

            measurements += {
                "samples": [
                    (
                        sample_time,
                        (
                            list(np.array(X).flatten()),
                            list(np.array(y).flatten()),
                        ),
                    )
                ]
            }

            self._platform.renode_run_for(1 / self._frequency)
            self._platform.inference_step_callback()

            if stop_event.is_set():
                break
