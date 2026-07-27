# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing decorators for benchmark data gathering.
"""

import json
import os
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from threading import Condition, Thread
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import psutil

from kenning.utils.logger import KLogger

try:
    from jtop import jtop
except ImportError:
    jtop = None

try:
    import pynvml
except ImportError:
    pynvml = None


class Measurements(object):
    """
    Stores benchmark measurements for later processing.

    This is a dict-like object that wraps all processing results for later
    report generation.

    The dictionary in Measurements has measurement type as a key, and list of
    values for given measurement type.

    There can be other values assigned to a given measurement type than list,
    but it requires explicit initialization.

    Attributes
    ----------
    data : dict
        Dictionary storing lists of values.
    """

    UNOPTIMIZED = "__unoptimized__"

    def __init__(self):
        self.data = dict()

    def __iadd__(self, other: Union[Dict, "Measurements"]) -> "Measurements":
        self.update_measurements(other)
        return self

    def initialize_measurement(self, measurement_type: str, value: Any):
        """
        Sets the initial value for a given measurement type.

        By default, the initial values for every measurement are empty lists.
        Lists are meant to collect time series data and other probed
        measurements for further analysis.

        In case the data is collected in a different container, it should
        be configured explicitly.

        Parameters
        ----------
        measurement_type : str
            The type (name) of the measurement.
        value : Any
            The initial value for the measurement type.
        """
        self.data[measurement_type] = value

    def update_measurements(self, other: Union[Dict, "Measurements"]):
        """
        Adds measurements of types given in the other object.

        It requires another Measurements object, or a dictionary that has
        string keys and values that are lists of values. The lists from the
        other object are appended to the lists in this object.

        Parameters
        ----------
        other : Union[Dict, 'Measurements']
            A dictionary or another Measurements object that contains lists in
            every entry.
        """
        assert isinstance(other, dict) or isinstance(other, Measurements)
        if isinstance(other, Measurements):
            for k, v in other.data.items():
                if k not in self.data:
                    self.data[k] = other.data[k]
                else:
                    self.data[k] += other.data[k]
        else:
            for k, v in other.items():
                if k not in self.data:
                    self.data[k] = other[k]
                else:
                    self.data[k] += other[k]

    def add_measurement(
        self,
        measurementtype: str,
        value: Any,
        initialvaluefunc: Callable[[], Any] = lambda: list(),
    ):
        """
        Add new value to a given measurement type.

        Parameters
        ----------
        measurementtype : str
            The measurement type to be updated.
        value : Any
            The value to add.
        initialvaluefunc : Callable[[], Any]
            The initial value for the measurement.
        """
        assert isinstance(measurementtype, str)
        if measurementtype not in self.data:
            self.data[measurementtype] = initialvaluefunc()
        self.data[measurementtype] += value

    def get_values(self, measurementtype: str) -> List:
        """
        Returns list of values for a given measurement type.

        Parameters
        ----------
        measurementtype : str
            The name of the measurement type.

        Returns
        -------
        List
            List of values for a given measurement type.
        """
        return self.data[measurementtype]

    def accumulate(
        self,
        measurementtype: str,
        valuetoadd: Any,
        initvaluefunc: Callable[[], Any] = lambda: 0,
    ):
        """
        Adds given value to a measurement.

        This function adds given value (it can be integer, float, numpy array,
        or any type that implements iadd operator).

        If it is the first assignment to a given measurement type, the first
        list element is initialized with the ``initvaluefunc`` (function
        returns the initial value).

        Parameters
        ----------
        measurementtype : str
            The name of the measurement.
        valuetoadd : Any
            New value to add to the measurement.
        initvaluefunc : Callable[[], Any]
            The initial value of the measurement, default 0.
        """
        if measurementtype not in self.data:
            self.data[measurementtype] = initvaluefunc()
        self.data[measurementtype] += valuetoadd

    def copy(self):
        """
        Makes copy of measurements data.
        """
        return self.data.copy()

    def clear(self):
        """
        Clears measurement data.
        """
        self.data.clear()


class MeasurementsCollector(object):
    """
    It is a 'static' class collecting measurements from various sources.
    """

    measurements = Measurements()

    @classmethod
    def set_unoptimized(
        cls,
        optimized_measurementspath: Path,
        unoptimized_measurementspath: Path,
        remove_unoptimized_measurementsfile: bool = True,
    ):
        """
        Copies unoptimized model measurements to `UNOPTIMIZED` field of the
        optimized model measurements.

        Parameters
        ----------
        optimized_measurementspath : Path
            Path to the optimized model measurements.
        unoptimized_measurementspath : Path
            Path to the unoptimized model measurements.
        remove_unoptimized_measurementsfile : bool
            Determines whether the unoptimized model measurements should be
            deleted.
        """
        with (
            open(optimized_measurementspath) as optimized_measurementsfile,
            open(unoptimized_measurementspath) as unoptimized_measurementsfile,
        ):
            optimized_measurements = json.load(optimized_measurementsfile)
            unoptimized_measurements = json.load(unoptimized_measurementsfile)

        optimized_measurements[
            Measurements.UNOPTIMIZED
        ] = unoptimized_measurements
        cls._dump(optimized_measurements, optimized_measurementspath)

        if remove_unoptimized_measurementsfile:
            unoptimized_measurementspath.unlink()

    @classmethod
    def save_measurements(cls, resultpath: Path):
        """
        Saves measurements to JSON file.

        Parameters
        ----------
        resultpath : Path
            Path to the saved JSON file.
        """
        for key, measurement in cls.measurements.data.items():
            if isinstance(measurement, np.ndarray):
                cls.measurements.data[key] = measurement.tolist()
        cls._dump(cls.measurements.data, resultpath)

    @staticmethod
    def _dump(measurementsdata: Dict, resultpath: Path):
        """
        Serializes measurements data into the given path.

        Parameters
        ----------
        measurementsdata : Dict
            Serializable measurements data
        resultpath : Path
            Path to the saved JSON file.
        """
        results_dir = Path(resultpath).parent
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
            KLogger.info(
                f"Created a directory for measurements: {results_dir}"
            )
        with open(resultpath, "w") as measurementsfile:
            json.dump(
                measurementsdata,
                measurementsfile,
                indent=2,
                default=str,
            )

    @classmethod
    def clear(cls):
        """
        Clears measurement data.
        """
        cls.measurements.clear()


def tagmeasurements(tagname: str) -> Callable:
    """
    Decorator for adding tags for measurements and saving their timestamps.

    Parameters
    ----------
    tagname : str
        The name of tag.

    Returns
    -------
    Callable
        Decorated function.
    """

    def statistics_decorator(function):
        @wraps(function)
        def statistics_wrapper(*args):
            starttimestamp = time.perf_counter()
            returnvalue = function(*args)
            endtimestamp = time.perf_counter()
            KLogger.debug(
                f"{function.__name__} start: {starttimestamp * 1000} ms end: "
                f"{endtimestamp * 1000} ms"
            )
            MeasurementsCollector.measurements += {
                "tags": [
                    {
                        "name": tagname,
                        "start": starttimestamp,
                        "end": endtimestamp,
                    }
                ]
            }
            return returnvalue

        return statistics_wrapper

    return statistics_decorator


def timemeasurements(
    measurementname: str,
    get_time_func: Callable[[], float] = time.perf_counter,
) -> Callable:
    """
    Decorator for measuring time of the function.

    The duration is given in nanoseconds.

    Parameters
    ----------
    measurementname : str
        The name of the measurement type.
    get_time_func : Callable[[], float]
        Function that returns current timestamp.

    Returns
    -------
    Callable
        Decorated function.
    """

    def statistics_decorator(function):
        @wraps(function)
        def statistics_wrapper(*args, **kwargs):
            start = get_time_func()
            returnvalue = function(*args, **kwargs)
            duration = get_time_func() - start
            KLogger.debug(f"{function.__name__} time:  {duration * 1000} ms")
            MeasurementsCollector.measurements += {
                measurementname: [duration],
                f"{measurementname}_timestamp": [get_time_func()],
            }
            return returnvalue

        return statistics_wrapper

    return statistics_decorator


class SystemStatsCollector(Thread):
    """
    It is a separate thread used for collecting system statistics.

    It collects:

    * CPU utilization,
    * RAM utilization,
    * GPU utilization,
    * GPU Memory utilization.

    It can be executed in parallel to another function to check its
    utilization of resources.
    """

    @dataclass
    class NvidiaGpuProcessInfo:
        """
        A dataclass to store NVIDIA GPU process statistics of Kenning.
        """

        gpu_index: int
        kenning_process: Any
        process_newest_usage: Any

    def __init__(self, prefix: str, step: float = 0.1):
        """
        Prepares thread for execution.

        Parameters
        ----------
        prefix : str
            The prefix used in measurements.
        step : float
            The step for the measurements, in seconds.
        """
        kenning_pid = os.getpid()
        Thread.__init__(self)
        self.measurements = Measurements()
        self.running = True
        self.prefix = prefix
        self.nvmlReady = False
        self.jetson = None
        self.step = step
        self.runningcondition = Condition()
        self.kenning_process = psutil.Process(kenning_pid)
        self.kenning_process.cpu_percent()
        self.last_nvidia_stats_read_us = 0
        self.nvidia_stats_read_window_sec = 2

        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.nvmlReady = True
            except Exception as ex:
                KLogger.warning(f"No NVML support due to error {ex}")

        if jtop is not None:
            try:
                self.jetson = jtop()
                self.jetson.start()
            except Exception as ex:
                KLogger.warning(f"No jtop support due to error {ex}")

        if (cpu_count := psutil.cpu_count()) is not None:
            self.cpu_count = cpu_count
        else:
            KLogger.warning(
                "Unknown number of CPU cores. Assuming core count 1."
            )
            self.cpu_count = 1

    def __enter__(self) -> "SystemStatsCollector":
        self.start()
        self.last_nvidia_stats_read_us = time.time_ns() // 1000 - 2 * 1_000_000
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        self.stop()
        self.join()
        return False

    def get_measurements(self) -> Measurements:
        """
        Returns measurements from the thread.

        Collected measurements names are prefixed by the prefix given in the
        constructor.

        The list of measurements:

        * `<prefix>_cpus_percent`: gives per-core CPU utilization (%),
        * `<prefix>_mem_percent`: gives overall memory usage (%),
        * `<prefix>_gpu_utilization`: gives overall GPU utilization (%),
        * `<prefix>_gpu_mem_utilization`: gives overall memory utilization (%),
        * `<prefix>_timestamp`: gives the timestamp of above measurements (ns).

        Returns
        -------
        Measurements
            Measurements object.
        """
        return self.measurements

    def run(self):
        self.measurements = Measurements()
        self.running = True

        while self.running:
            self._get_process_cpu_stats()

            if self.jetson and self.jetson.ok(spin=True):
                self._get_global_jetson_stats()
            elif pynvml and self.nvmlReady:
                gpu_processes = self._get_nvidia_gpu_processes()
                if not gpu_processes:
                    KLogger.debug(
                        "No NVIDIA GPU present in system."
                        " GPU logs will not be collected."
                    )
                else:
                    stats = self._is_kenning_process_in_nvidia(gpu_processes)
                    if stats is not None:
                        self._get_process_nvidia_stats(stats)
                    else:
                        # TODO when Kenning report will support multiple GPU
                        # usage reports, change to global GPU load
                        self._get_global_nvidia_stats(
                            next(iter(gpu_processes))
                        )

            with self.runningcondition:
                self.runningcondition.wait(timeout=self.step)

    def _get_nvidia_gpu_processes(self) -> Dict[int, List]:
        gpu_processes = {}
        if not pynvml:
            return gpu_processes

        try:
            for gpu_index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                processes = [
                    *pynvml.nvmlDeviceGetComputeRunningProcesses(handle),
                    *pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle),
                    *pynvml.nvmlDeviceGetMPSComputeRunningProcesses(handle),
                ]
                gpu_processes[gpu_index] = processes
        except pynvml.NVMLError as ex:
            KLogger.debug(
                "Failed to get running processes from NVIDIA GPU,"
                f" traceback is: \n{ex}"
            )

        return gpu_processes

    def _is_kenning_process_in_nvidia(
        self, gpu_processes: Dict[int, List[Any]]
    ) -> Optional[NvidiaGpuProcessInfo]:
        if not pynvml:
            return None

        kenning_gpu_index = -1
        kenning_process = None
        for gpu_index, processes in gpu_processes.items():
            for process in processes:
                if self.kenning_process.pid == process.pid:
                    kenning_process = process
                    kenning_gpu_index = gpu_index
                    break

            if kenning_gpu_index != -1:
                break

        if kenning_gpu_index == -1:
            KLogger.debug("No Kenning process in GPU")
            return None

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(kenning_gpu_index)
            samples = pynvml.nvmlDeviceGetProcessUtilization(
                handle, self.last_nvidia_stats_read_us
            )
        except pynvml.NVMLError as ex:
            KLogger.error(
                "Failed to get process utilization from NVIDIA GPU,"
                f" traceback is:\n{ex}"
            )
            return None

        self.last_nvidia_stats_read_us = (
            time.time_ns() - self.nvidia_stats_read_window_sec * 10**9
        ) // 1000

        process_newest_usage = None
        for sample in samples:
            if sample.pid != self.kenning_process.pid:
                continue

            if (
                process_newest_usage is None
                or sample.timeStamp > process_newest_usage.timeStamp
            ):
                process_newest_usage = sample

        if process_newest_usage:
            self.process_present_in_nvidia_gpu = True
            return self.NvidiaGpuProcessInfo(
                gpu_index=gpu_index,
                kenning_process=kenning_process,
                process_newest_usage=process_newest_usage,
            )
        else:
            return None

    def _get_process_nvidia_stats(self, stats: NvidiaGpuProcessInfo) -> None:
        if not pynvml:
            return

        if (used_vram := stats.kenning_process.usedGpuMemory) is None:
            KLogger.debug(
                "GPU has not reported total memory."
                " Usage percent calculation is not possible."
            )
            used_vram = 0

        gpu_usage_percent = None
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(stats.gpu_index)
            total_vram = pynvml.nvmlDeviceGetMemoryInfo(handle).total

            if total_vram is not None and total_vram != 0:
                gpu_usage_percent = used_vram / int(total_vram) * 100
        except pynvml.NVMLError as ex:
            KLogger.error(
                "Failed to get VRAM information from NVIDIA GPU,"
                f" traceback is:\n{ex}"
            )

        self.measurements += {
            f"{self.prefix}_gpu_utilization": [
                stats.process_newest_usage.smUtil
            ],
            f"{self.prefix}_gpu_mem_utilization": [gpu_usage_percent],
            f"{self.prefix}_gpu_timestamp": [time.perf_counter()],
        }

    def _get_global_nvidia_stats(self, gpu_index: int) -> None:
        if not pynvml:
            return

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            vram_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        except pynvml.NVMLError as ex:
            KLogger.error(
                "Failed to get statistics from NVIDIA GPU,"
                f" traceback is:\n{ex}"
            )

            self.measurements += {
                f"{self.prefix}_gpu_utilization": [None],
                f"{self.prefix}_gpu_mem_utilization": [None],
                f"{self.prefix}_gpu_timestamp": [time.perf_counter()],
            }
            return

        gpu_usage_percent = (
            vram_info.used / vram_info.total * 100 if vram_info.total else None
        )

        self.measurements += {
            f"{self.prefix}_gpu_utilization": [util.gpu],
            f"{self.prefix}_gpu_mem_utilization": [gpu_usage_percent],
            f"{self.prefix}_gpu_timestamp": [time.perf_counter()],
        }

    def _get_process_cpu_stats(self) -> None:
        if (cpu_percent := self.kenning_process.cpu_percent()) is None:
            KLogger.warning(
                "Unknown Kenning process usage, assuming 0 percent."
            )
            cpu_percent = 0

        cpus = cpu_percent / self.cpu_count
        memory_percent = self.kenning_process.memory_percent()
        self.measurements += {
            f"{self.prefix}_cpus_percent": [[cpus]],
            f"{self.prefix}_mem_percent": [memory_percent],
            f"{self.prefix}_timestamp": [time.perf_counter()],
        }

    def _get_global_jetson_stats(self) -> None:
        if not self.jetson:
            return

        gpu = self.jetson.stats
        if gpu and "GPU" in gpu:
            gpu = self.jetson.stats
            # It is unable to get Jetson GPU load on given process
            gpu_usage = float(gpu["GPU"])
            memory_percent = self.kenning_process.memory_percent()

            self.measurements += {
                f"{self.prefix}_gpu_utilization": [gpu_usage],
                # Jetson use RAM for GPU memory
                f"{self.prefix}_gpu_mem_utilization": [memory_percent],
                f"{self.prefix}_gpu_timestamp": [time.perf_counter()],
            }

        # collect power information from each lines:
        if hasattr(self.jetson, "power"):
            power = self.jetson.power

            if "rail" in power:
                rails = power["rail"]

                for name, stats in rails.items():
                    voltage = float(stats["volt"])
                    current = float(stats["curr"])
                    power = float(stats["power"])

                    name = name.lower()
                    self.measurements += {
                        f"{self.prefix}_{name}_voltage": [voltage],
                        f"{self.prefix}_{name}_current": [current],
                        f"{self.prefix}_{name}_power": [power],
                    }

        # collect frequency information from each engine:
        if hasattr(self.jetson, "engine"):
            engines = self.jetson.engine

            for group in engines.keys():
                for name, engine in engines[group].items():
                    frequency = float(engine["cur"])
                    self.measurements += {
                        f"{self.prefix}_{name.lower()}_frequency": [frequency]
                    }

    def stop(self):
        self.running = False
        if self.jetson:
            self.jetson.close()

        if pynvml and self.nvmlReady:
            pynvml.nvmlShutdown()

        with self.runningcondition:
            self.runningcondition.notify_all()


def systemstatsmeasurements(
    measurementname: str, step: float = 0.5
) -> Callable:
    """
    Decorator for measuring memory usage of the function.

    Check SystemStatsCollector.get_measurements for list of delivered
    measurements.

    Parameters
    ----------
    measurementname : str
        The name of the measurement type.
    step : float
        The step for the measurements, in seconds.

    Returns
    -------
    Callable
        Decorated function.
    """

    def statistics_decorator(function):
        @wraps(function)
        def statistics_wrapper(*args):
            with SystemStatsCollector(
                measurementname, step
            ) as measurementsthread:
                returnvalue = function(*args)
                MeasurementsCollector.measurements += (
                    measurementsthread.get_measurements()
                )
            return returnvalue

        return statistics_wrapper

    return statistics_decorator
