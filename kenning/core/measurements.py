# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing decorators for benchmark data gathering.
"""

from typing import List, Dict, Union, Any, Callable
import time
from kenning.utils import logger
import psutil
import subprocess
import re
import numpy as np
from pathlib import Path
import json
import tempfile

try:
    from pynvml.smi import nvidia_smi
except ImportError:
    nvidia_smi = None
from threading import Thread, Condition

from shutil import which

from functools import wraps

logger = logger.get_logger()


is_nvidia_smi_loadable = True


class Measurements(object):
    """
    Stores benchmark measurements for later processing.

    This is a dict-like object that wraps all processing results for later
    raport generation.

    The dictionary in Measurements has measurement type as a key, and list of
    values for given measurement type.

    There can be other values assigned to a given measurement type than list,
    but it requires explicit initialization.

    Attributes
    ----------
    data : dict
        Dictionary storing lists of values
    """
    def __init__(self):
        self.data = dict()

    def __iadd__(self, other: Union[Dict, 'Measurements']) -> 'Measurements':
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
            The type (name) of the measurement
        value : Any
            The initial value for the measurement type
        """
        self.data[measurement_type] = value

    def update_measurements(self, other: Union[Dict, 'Measurements']):
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

    def add_measurements_list(
            self,
            measurementtype: str,
            valueslist: List):
        """
        Adds new values to a given measurement type.

        Parameters
        ----------
        measurementtype : str
            the measurement type to be updated
        valueslist : List
            the list of values to add
        """
        assert isinstance(valueslist, list)
        assert isinstance(measurementtype, str)
        if measurementtype not in self.data:
            self.data[measurementtype] = list()
        self.data[measurementtype] += valueslist

    def add_measurement(
            self,
            measurementtype: str,
            value: Any,
            initialvaluefunc: Callable[[], Any] = lambda: list()):
        """
        Add new value to a given measurement type.

        Parameters
        ----------
        measurementtype : str
            the measurement type to be updated
        value : Any
            the value to add
        initialvaluefunc : Callable
            the initial value for the measurement
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
            The name of the measurement type

        Returns
        -------
        List : list of values for a given measurement type
        """
        return self.data[measurementtype]

    def accumulate(
            self,
            measurementtype: str,
            valuetoadd: Any,
            initvaluefunc: Callable[[], Any] = lambda: 0) -> List:
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
            the name of the measurement
        valuetoadd : Any
            New value to add to the measurement
        initvaluefunc : Callable[[], Any]
            The initial value of the measurement, default 0
        """
        if measurementtype not in self.data:
            self.data[measurementtype] = initvaluefunc()
        self.data[measurementtype] += valuetoadd

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
    def save_measurements(cls, resultpath: Path):
        """
        Saves measurements to JSON file.

        Parameters
        ----------
        resultpath : Path
            Path to the saved JSON file
        """
        if 'eval_confusion_matrix' in cls.measurements.data:
            cls.measurements.data['eval_confusion_matrix'] = cls.measurements.data['eval_confusion_matrix'].tolist()  # noqa: E501
        with open(resultpath, 'w') as measurementsfile:
            json.dump(
                cls.measurements.data,
                measurementsfile,
                indent=2
            )

    @classmethod
    def clear(cls):
        """
        Clears measurement data.
        """
        cls.measurements.clear()


def tagmeasurements(tagname: str):
    """
    Decorator for adding tags for measurements and saving their timestamps.

    Parameters
    ----------
    tagname: str
        The name of tag.
    """
    def statistics_decorator(function):
        @wraps(function)
        def statistics_wrapper(*args):
            starttimestamp = time.perf_counter()
            returnvalue = function(*args)
            endtimestamp = time.perf_counter()
            logger.debug(
                f'{function.__name__} start: {starttimestamp * 1000} ms end: {endtimestamp * 1000} ms'  # noqa: E501
            )
            MeasurementsCollector.measurements += {
                'tags': [
                    {'name': tagname, 'start': starttimestamp, 'end': endtimestamp}  # noqa: E501
                ]
            }
            return returnvalue
        return statistics_wrapper
    return statistics_decorator


def timemeasurements(measurementname: str):
    """
    Decorator for measuring time of the function.

    The duration is given in nanoseconds.

    Parameters
    ----------
    measurementname : str
        The name of the measurement type.
    """
    def statistics_decorator(function):
        @wraps(function)
        def statistics_wrapper(*args):
            start = time.perf_counter()
            returnvalue = function(*args)
            duration = time.perf_counter() - start
            logger.debug(
                f'{function.__name__} time:  {duration * 1000} ms'
            )
            MeasurementsCollector.measurements += {
                measurementname: [duration],
                f'{measurementname}_timestamp': [time.perf_counter()]
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
    def __init__(self, prefix: str, step: float = 0.1):
        """
        Prepares thread for execution.

        Parameters
        ----------
        prefix : str
            The prefix used in measurements
        step : float
            The step for the measurements, in seconds
        """
        global is_nvidia_smi_loadable
        Thread.__init__(self)
        self.measurements = Measurements()
        self.running = True
        self.prefix = prefix
        if is_nvidia_smi_loadable and nvidia_smi is not None:
            try:
                self.nvidia_smi = nvidia_smi.getInstance()
            except Exception as ex:
                logger.warning(
                    f'No NVML support due to error {ex}'
                )
                self.nvidia_smi = None
                is_nvidia_smi_loadable = False
        else:
            self.nvidia_smi = None
        self.step = step
        self.runningcondition = Condition()

    def get_measurements(self):
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
        Measurements : Measurements object.
        """
        return self.measurements

    def run(self):
        self.measurements = Measurements()
        self.running = True
        tegrastatsoutputfd = None
        try:
            tegrastats = which('tegrastats')
            if tegrastats is not None:
                tegrastatsoutput = tempfile.NamedTemporaryFile()
                tegrastatsoutputfd = open(tegrastatsoutput.name, 'w')
                tegrastatsstart = time.perf_counter()
                tegrastatsproc = subprocess.Popen(
                    f'{tegrastats} --interval {self.step * 1000}'.split(' '),
                    stdout=tegrastatsoutputfd
                )
            while self.running:
                cpus = psutil.cpu_percent(interval=0, percpu=True)
                mem = psutil.virtual_memory()
                self.measurements += {
                    f'{self.prefix}_cpus_percent': [cpus],
                    f'{self.prefix}_mem_percent': [mem.percent],
                    f'{self.prefix}_timestamp': [time.perf_counter()],
                }
                if self.nvidia_smi is not None:
                    gpu = self.nvidia_smi.DeviceQuery(
                        'memory.free, memory.total, utilization.gpu'
                    )
                    memtot = float(gpu['gpu'][0]['fb_memory_usage']['total'])
                    memfree = float(gpu['gpu'][0]['fb_memory_usage']['free'])
                    gpumemutilization = (memtot - memfree) / memtot * 100.0
                    gpuutilization = float(
                        gpu['gpu'][0]['utilization']['gpu_util']
                    )
                    self.measurements += {
                        f'{self.prefix}_gpu_utilization': [gpuutilization],
                        f'{self.prefix}_gpu_mem_utilization': [
                            gpumemutilization
                        ],
                        f'{self.prefix}_gpu_timestamp': [time.perf_counter()],
                    }
                with self.runningcondition:
                    self.runningcondition.wait(timeout=self.step)
            if tegrastats:
                tegrastatsproc.terminate()
                tegrastatsend = time.perf_counter()
                tegrastatsoutputfd.close()
                with open(tegrastatsoutput.name, 'r') as tegrastatsoutputfd:
                    readings = tegrastatsoutputfd.read().split('\n')
                tegrastatsoutputfd = None
                ramusages = []
                gpuutilization = []
                vdd_gpu_soc = []
                vdd_cpu_cv = []
                vin_sys_5v0 = []
                vddq_vdd2_1v8ao = []
                cpupower = []
                gpupower = []
                socpower = []
                cvpower = []
                vddrqpower = []
                sys5vpower = []
                for entry in readings:
                    match = re.match(r'.*RAM (\d+)/(\d+)MB', entry)
                    if match:
                        currram = float(match.group(1))
                        totram = float(match.group(2))
                        ramusages.append(int(currram / totram * 100))
                    match = re.match(r'.*GR3D_FREQ (\d+)%', entry)
                    if match:
                        gpuutilization.append(int(match.group(1)))
                    match = re.match(r'.*VDD_GPU_SOC (\d+)mW/(\d+)mW', entry)
                    if match:
                        vdd_gpu_soc.append(int(match.group(1)))
                    match = re.match(r'.*VDD_CPU_CV (\d+)mW/(\d+)mW', entry)
                    if match:
                        vdd_cpu_cv.append(int(match.group(1)))
                    match = re.match(r'.*VIN_SYS_5V0 (\d+)mW/(\d+)mW', entry)
                    if match:
                        vin_sys_5v0.append(int(match.group(1)))
                    match = re.match(
                        r'.*VDDQ_VDD2_1V8AO (\d+)mW/(\d+)mW',
                        entry
                    )
                    if match:
                        vddq_vdd2_1v8ao.append(int(match.group(1)))
                    match = re.match(r'.*CPU (\d+)mW/(\d+)mW', entry)
                    if match:
                        cpupower.append(int(match.group(1)))
                    match = re.match(r'.*GPU (\d+)mW/(\d+)mW', entry)
                    if match:
                        gpupower.append(int(match.group(1)))
                    match = re.match(r'.*SOC (\d+)mW/(\d+)mW', entry)
                    if match:
                        socpower.append(int(match.group(1)))
                    match = re.match(r'.*CV (\d+)mW/(\d+)mW', entry)
                    if match:
                        cvpower.append(int(match.group(1)))
                    match = re.match(r'.*VDDRQ (\d+)mW/(\d+)mW', entry)
                    if match:
                        vddrqpower.append(int(match.group(1)))
                    match = re.match(r'.*SYS5V (\d+)mW/(\d+)mW', entry)
                    if match:
                        sys5vpower.append(int(match.group(1)))
                timestamps = np.linspace(
                    tegrastatsstart,
                    tegrastatsend,
                    num=len(readings)-1,
                    endpoint=True
                ).tolist()
                self.measurements += {
                    f'{self.prefix}_gpu_utilization': gpuutilization,
                    f'{self.prefix}_gpu_mem_utilization': ramusages,
                    f'{self.prefix}_power_vdd_gpu_soc': vdd_gpu_soc,
                    f'{self.prefix}_power_vdd_cpu_cv': vdd_cpu_cv,
                    f'{self.prefix}_power_vin_sys_5v0': vin_sys_5v0,
                    f'{self.prefix}_power_vddq_vdd2_1v8ao': vddq_vdd2_1v8ao,
                    f'{self.prefix}_power_cpu': cpupower,
                    f'{self.prefix}_power_gpu': gpupower,
                    f'{self.prefix}_power_soc': socpower,
                    f'{self.prefix}_power_cv': cvpower,
                    f'{self.prefix}_power_vddrq': vddrqpower,
                    f'{self.prefix}_power_sys5v': sys5vpower,
                    f'{self.prefix}_gpu_timestamp': timestamps
                }
        finally:
            if tegrastatsoutputfd:
                tegrastatsoutputfd.close()

    def stop(self):
        self.running = False
        with self.runningcondition:
            self.runningcondition.notify_all()


def systemstatsmeasurements(measurementname: str, step: float = 0.5):
    """
    Decorator for measuring memory usage of the function.

    Check SystemStatsCollector.get_measurements for list of delivered
    measurements.

    Parameters
    ----------
    measurementname : str
        The name of the measurement type.
    step : float
        The step for the measurements, in seconds
    """

    def statistics_decorator(function):
        @wraps(function)
        def statistics_wrapper(*args):
            measurementsthread = SystemStatsCollector(measurementname, step)
            measurementsthread.start()
            returnvalue = function(*args)
            measurementsthread.stop()
            measurementsthread.join()
            MeasurementsCollector.measurements += \
                measurementsthread.get_measurements()
            return returnvalue
        return statistics_wrapper
    return statistics_decorator
