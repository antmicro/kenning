# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing classes related to Renode sensors.
"""

from abc import ABC, abstractmethod


class Sensor(ABC):
    """
    Abstract wrapper for Renode sensors.
    """

    sensor_map = {}

    def __init__(self, peripheral):
        self._peripheral = peripheral

    def __init_subclass__(cls, sensor_name, **kwargs):
        cls.sensor_map[sensor_name] = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def create_from_peripheral(cls, peripheral):
        """
        Creates an instance of an appropriate Sensor subclass
        for a given pyrenode3 peripheral.
        """
        sensor_name = peripheral.internal.__class__.__name__
        sensor_cls = cls.sensor_map[sensor_name]
        return sensor_cls(peripheral)

    @abstractmethod
    def feed(self, data: list[float]):
        """
        Feed data that should be returned during sensor reads.
        """
        ...

    @abstractmethod
    def size(self):
        """
        Total number of sensor channels passed as data to `feed`.
        """
        ...


class LIS2DS12Sensor(Sensor, sensor_name="LIS2DS12"):
    """
    Wrapper for LIS2DS12 Renode sensor.
    """

    def feed(self, data):
        from System import Decimal

        X, Y, Z = [Decimal(x) for x in data]
        self._peripheral.AccelerationX = X
        self._peripheral.AccelerationY = Y
        self._peripheral.AccelerationZ = Z

    def size(self):
        return 3


class ADXL345Sensor(Sensor, sensor_name="ADXL345"):
    """
    Wrapper for ADXL345 Renode sensor.
    """

    def feed(self, data):
        X, Y, Z = list(map(lambda x: int(x * 1000), data))
        self._peripheral.FeedSample(X, Y, Z, -1)

    def size(self):
        return 3
