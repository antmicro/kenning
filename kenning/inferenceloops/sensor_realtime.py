# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing a sensor implementation of the real-time inference loop.
"""

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
        self._measurements = None

    def _prepare(self):
        for sensor_path in self._platform.sensors:
            sensor = reduce(
                lambda x, y: getattr(x, y),
                sensor_path.split("."),
                self._platform.machine.sysbus,
            )
            self._sensors.append(Sensor.create_from_peripheral(sensor))

        self._dataset_last_sample_time = self._platform.get_time()
        self._dataset_iter = iter(list(self._dataset.iter_test()))

        def hook(a, _):
            cur_time = self._platform.get_time()
            if cur_time - self._dataset_last_sample_time > 1 / self._frequency:
                self._dataset_last_sample_time = cur_time
                sample = next(self._dataset_iter, None)
                if sample is None:
                    if self._stop_event is not None:
                        self._stop_event.set()
                    return a
                if self._measurements is not None:
                    self._feed_sample(sample, self._measurements)
            return a

        self._set_i2c_hook(hook)

    def _pre_loop_hook(self, measurements):
        self._measurements = measurements

    def _set_i2c_hook(self, f):
        import System

        for width in [
            System.Byte,
            System.UInt16,
            System.UInt32,
            System.UInt64,
        ]:
            hook = System.Func[width, System.Int64, width](f)
            i2c1 = self._platform.machine.sysbus.i2c1.internal
            self._platform.machine.sysbus.internal.SetHookAfterPeripheralRead[
                width
            ](i2c1, hook)
            self._platform.machine.sysbus.internal.SetHookBeforePeripheralWrite[
                width
            ](i2c1, hook)

    def _feed_sample(self, sample, measurements):
        X, y = sample
        if getattr(self._dataset, "window_size", None):
            X = list(map(float, X[0][0][-1]))
        else:
            X = list(map(float, X[0][0]))

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

    def _feed_data(self, measurements: Measurements):
        while not self._stop_event.is_set():
            self._stop_event.wait(0.1)
