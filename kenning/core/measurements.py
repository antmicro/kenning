# Copyright (c) 2020-2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing decorators for benchmark data gathering.
"""

import json
import os
import sys
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

try:
    import amdsmi
except ImportError:
    amdsmi = None

try:
    rocm_smi_path = Path("/opt/rocm/libexec/rocm_smi/")
    if rocm_smi_path.exists():
        sys.path.append(str(rocm_smi_path))
    import rocm_smi
except ImportError:
    rocm_smi = None


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

    def __init__(self, prefix: str, step: float = 0.1, pid: int = os.getpid()):
        """
        Prepares thread for execution.

        Parameters
        ----------
        prefix : str
            The prefix used in measurements.
        step : float
            The step for the measurements, in seconds.
        pid : int
            Optional pid of the process that should be monitored.
        """
        kenning_pid = pid
        Thread.__init__(self)
        self.measurements = Measurements()
        self.running = True
        self.prefix = prefix
        self.nvmlReady = False
        self.jetson = None

        if amdsmi is not None:
            self.amdsmi_enabled = self._amdsmi_enable()
        else:
            self.amdsmi_enabled = False

        if rocm_smi is not None and not self.amdsmi_enabled:
            KLogger.info("Attempting to use ROCM SMI as fallback for AMD SMI")
            self.rocm_smi_enabled = self._rocm_smi_enable()
        else:
            self.rocm_smi_enabled = False

        self.step = step
        self.runningcondition = Condition()
        self.kenning_process = psutil.Process(kenning_pid)
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

    def _amdsmi_enable(self) -> bool:
        """
        Initializes AMD SMI and performs test data read.
        """
        # Initialize AMD SMI and check available GPUs
        try:
            amdsmi.amdsmi_init()
            amdsmi_handles = amdsmi.amdsmi_get_processor_handles()
            self.amdsmi_handle = amdsmi_handles[0]
        except amdsmi.AmdSmiException as ex:
            KLogger.warning(f"Failed to initialize AMD SMI: {ex}")
            self._amdsmi_shut_down()
            return False
        except IndexError:
            KLogger.warning("No AMD SMI compatible GPUs found.")
            self._amdsmi_shut_down()
            return False

        self.pid = os.getpid()

        # Attempt to obtain information about current process gpu usage
        try:
            info = amdsmi.amdsmi_get_gpu_compute_process_info_by_pid(self.pid)
            vram_info = amdsmi.amdsmi_get_gpu_vram_usage(self.amdsmi_handle)
        except amdsmi.AmdSmiException as ex:
            KLogger.warning(
                f"Failed to obtain test information from AMD SMI: {ex}"
            )
            self._amdsmi_shut_down()
            return False

        if (
            "vram_total" not in vram_info
            or type(vram_info["vram_total"]) is not int
        ):
            KLogger.warning(
                "Failed to obtain total amount of vram from AMD SMI"
            )
            self._amdsmi_shut_down()
            return False

        self.amdsmi_total_vram = vram_info["vram_total"]

        # If AMD SMI doesn't support fully our GPU it can return partial data
        # with some values equal to 'N\A' instead of int.
        if (
            type(info["vram_usage"]) is not int
            or not (0 <= info["vram_usage"] <= self.amdsmi_total_vram)
            or type(info["cu_occupancy"]) is not int
            or not (0 <= info["cu_occupancy"] <= 100)
        ):
            KLogger.warning("Failed to parse test information from AMD SMI")
            self._amdsmi_shut_down()
            return False

        KLogger.info("AMD SMI support successfully enabled")
        return True

    def _amdsmi_shut_down(self):
        """
        Safe wrapper for amdsmi.amdsmi_shut_down.
        """
        try:
            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiException as ex:
            KLogger.warning(f"Failed to shut down AMD SMI: {ex}")

    def _rocm_smi_enable(self) -> bool:
        try:
            rocm_smi.initializeRsmi()
            devices = rocm_smi.listDevices()
            self.rocm_device = devices[0]
            rocm_smi.getAllocatedMemoryPercent(self.rocm_device)
            rocm_smi.getGpuUse(self.rocm_device)
        except IndexError:
            KLogger.warning("No ROCM SMI compatible GPU found.")
        except Exception as ex:
            KLogger.warning(f"Failed to initialize ROCM SMI: {ex}")
        else:
            KLogger.info("ROCM SMI support successfully enabled")
            return True

        return False

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

    def __del__(self):
        if self.amdsmi_enabled:
            self._amdsmi_shut_down()

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
            self._get_system_cpu_stats()

            if self.jetson and self.jetson.ok(spin=True):
                self._get_system_jetson_stats()
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

                    # TODO when Kenning report will support multiple GPU
                    # usage reports, change to system GPU load
                    self._get_system_nvidia_stats(next(iter(gpu_processes)))

            elif self.amdsmi_enabled:
                try:
                    info = amdsmi.amdsmi_get_gpu_compute_process_info_by_pid(
                        self.pid
                    )
                except amdsmi.AmdSmiException as ex:
                    KLogger.warning(f"Failed to fetch data from AMD SMI: {ex}")
                else:
                    self.measurements += {
                        f"{self.prefix}_process_gpu_mem_percent": [
                            info["vram_usage"] * 100 // self.amdsmi_total_vram
                        ],
                        f"{self.prefix}_process_gpu_percent": [
                            info["cu_occupancy"]
                        ],
                        f"{self.prefix}_process_gpu_timestamp": [
                            time.perf_counter()
                        ],
                    }
            elif self.rocm_smi_enabled:
                try:
                    vram_usage = rocm_smi.getAllocatedMemoryPercent(
                        self.rocm_device
                    )
                    gpu_usage = rocm_smi.getGpuUse(self.rocm_device)
                except Exception as ex:
                    KLogger.warning(
                        f"Failed to fetch data from ROCM SMI: {ex}"
                    )
                else:
                    self.measurements += {
                        f"{self.prefix}_system_gpu_mem_percent": [
                            vram_usage["value"],
                        ],
                        f"{self.prefix}_system_gpu_percent": [
                            gpu_usage,
                        ],
                        f"{self.prefix}_system_gpu_timestamp": [
                            time.perf_counter()
                        ],
                    }

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
            if not getattr(self, "_logged_no_gpu_in_process", False):
                KLogger.debug("No Kenning process in GPU")
                self._logged_no_gpu_in_process = True
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
            f"{self.prefix}_process_gpu_percent": [
                stats.process_newest_usage.smUtil
            ],
            f"{self.prefix}_process_gpu_mem_percent": [gpu_usage_percent],
            f"{self.prefix}_process_gpu_timestamp": [time.perf_counter()],
        }

    def _get_system_nvidia_stats(self, gpu_index: int) -> None:
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
                f"{self.prefix}_system_gpu_percent": [None],
                f"{self.prefix}_system_gpu_mem_percent": [None],
                f"{self.prefix}_system_gpu_timestamp": [time.perf_counter()],
            }
            return

        gpu_usage_percent = (
            vram_info.used / vram_info.total * 100 if vram_info.total else None
        )

        self.measurements += {
            f"{self.prefix}_system_gpu_percent": [util.gpu],
            f"{self.prefix}_system_gpu_mem_percent": [gpu_usage_percent],
            f"{self.prefix}_system_gpu_timestamp": [time.perf_counter()],
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
            # `cpus` for process scope has type `float` and is already average
            f"{self.prefix}_process_cpu_percent": [cpus],
            f"{self.prefix}_process_cpu_mem_percent": [memory_percent],
            f"{self.prefix}_process_cpu_timestamp": [time.perf_counter()],
        }

    def _get_system_cpu_stats(self) -> None:
        cpus = psutil.cpu_percent(interval=0, percpu=True)
        mem = psutil.virtual_memory()
        self.measurements += {
            # not a typo, `cpus` has type `list[float]` and mean value
            # is later calculated into `"{self.prefix}_system_cpu_percent"`
            f"{self.prefix}_system_cpu_percents": [cpus],
            f"{self.prefix}_system_cpu_mem_percent": [mem.percent],
            f"{self.prefix}_system_cpu_timestamp": [time.perf_counter()],
        }

    def _get_system_jetson_stats(self) -> None:
        if not self.jetson:
            return

        gpu = self.jetson.stats
        if gpu and "GPU" in gpu:
            gpu = self.jetson.stats
            # It is unable to get Jetson GPU load on given process
            gpu_usage = float(gpu["GPU"])
            memory_percent = self.kenning_process.memory_percent()

            self.measurements += {
                f"{self.prefix}_system_gpu_percent": [gpu_usage],
                # Jetson use RAM for GPU memory
                f"{self.prefix}_system_gpu_mem_percent": [memory_percent],
                f"{self.prefix}_system_gpu_timestamp": [time.perf_counter()],
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


class RemoteSystemStatsCollector(Thread):
    """
    It is a separate thread used for collecting system statistics. From a
    remote machine.

    It collects, if provided:

    * CPU utilization,
    * RAM utilization,
    * GPU utilization,
    * GPU Memory utilization.

    It can be executed in parallel to another function to check its
    utilization of resources on the remote machine.
    """

    def __init__(
        self,
        prefix: str,
        protocol: Any,
        step: float = 0.1,
    ):
        """
        Prepares thread for execution.

        Parameters
        ----------
        prefix : str
            The prefix used in measurements.
        protocol : Any
            Protocol used to fetch statistics.
        step : float
            The step for the measurements, in seconds.
        """
        Thread.__init__(self)
        self.measurements = Measurements()
        self.running = True
        self.prefix = prefix
        self.step = step
        self.runningcondition = Condition()
        self.protocol = protocol

    def __enter__(self) -> "SystemStatsCollector":
        self.protocol.initialize_client()
        self.start()
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
        Returns measurements from the remote machine.

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
            remote_meas = self.protocol.download_statistics(True)
            self.measurements += remote_meas
            with self.runningcondition:
                self.runningcondition.wait(timeout=self.step)

    def stop(self):
        self.running = False
        with self.runningcondition:
            self.runningcondition.notify_all()
        self.protocol.disconnect()


def serverstatsmeasurements(
    measurementname: str, host: str, port: int = 4371, step: float = 1.0
) -> Callable:
    """
    Decorator for measuring memory usage of the function.

    Check SystemStatsCollector.get_measurements for list of delivered
    measurements.

    Parameters
    ----------
    measurementname : str
        The name of the measurement type.
    host : str
        Host to connect to to get measurements.
    port : int
        Port on which measurements will be served
    step : float
        The step for the measurements, in seconds.

    Returns
    -------
    Callable
        Decorated function.
    """
    from kenning.protocols.network import NetworkProtocol

    def statistics_decorator(function):
        @wraps(function)
        def statistics_wrapper(*args):
            protocol_args = {
                "host": host,
                "port": port,
            }
            with RemoteSystemStatsCollector(
                measurementname,
                NetworkProtocol(**protocol_args),
                step,
            ) as measurementsthread:
                returnvalue = function(*args)
                MeasurementsCollector.measurements += (
                    measurementsthread.get_measurements()
                )
            return returnvalue

        return statistics_wrapper

    return statistics_decorator
