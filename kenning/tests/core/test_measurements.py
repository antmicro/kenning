# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Callable, List

import pytest

from kenning.core.measurements import (
    Measurements,
    MeasurementsCollector,
    tagmeasurements,
    timemeasurements,
)


def fake_clock(timestamps: List[float]) -> Callable[[], float]:
    """
    Returns a function which yields consecutive timestamps from the list.

    Parameters
    ----------
    timestamps : List[float]
        Timestamps to be returned by the consecutive calls.

    Returns
    -------
    Callable[[], float]
        Deterministic replacement for the time measuring function.
    """
    iterator = iter(timestamps)
    return lambda: next(iterator)


class TestMeasurements:
    def test_initialize_measurement_sets_non_list_container(self):
        """
        Tests that a measurement can be explicitly initialized with
        a container other than a list.
        """
        measurements = Measurements()

        measurements.initialize_measurement("counter", 5)

        assert measurements.data["counter"] == 5

    def test_add_measurement_creates_missing_entry(self):
        """
        Tests that adding a value to an unknown measurement type creates
        the entry using the default initial value.
        """
        measurements = Measurements()

        measurements.add_measurement("inference_step", [1.0])

        assert measurements.data["inference_step"] == [1.0]

    def test_add_measurement_appends_to_existing_entry(self):
        """
        Tests that consecutive values are appended to the already
        existing measurement.
        """
        measurements = Measurements()

        measurements.add_measurement("inference_step", [1.0])
        measurements.add_measurement("inference_step", [2.0])

        assert measurements.data["inference_step"] == [1.0, 2.0]

    def test_add_measurement_uses_custom_initial_value(self):
        """
        Tests that the provided initial value function is used when the
        measurement is created.
        """
        measurements = Measurements()

        measurements.add_measurement("counter", 2, initialvaluefunc=lambda: 10)

        assert measurements.data["counter"] == 12

    def test_accumulate_sums_values(self):
        """
        Tests that accumulate creates a scalar entry and adds consecutive
        values to it.
        """
        measurements = Measurements()

        measurements.accumulate("total", 2)
        measurements.accumulate("total", 3)

        assert measurements.data["total"] == 5

    def test_get_values_returns_collected_values(self):
        """
        Tests that get_values returns all values of a measurement type.
        """
        measurements = Measurements()
        measurements.add_measurement("inference_step", [1.0, 2.0])

        assert measurements.get_values("inference_step") == [1.0, 2.0]

    def test_get_values_raises_for_unknown_type(self):
        """
        Tests that reading an unknown measurement type is an error.
        """
        measurements = Measurements()

        with pytest.raises(KeyError):
            measurements.get_values("inference_step")

    def test_update_measurements_merges_dict(self):
        """
        Tests that a dictionary is merged into the measurements, with
        new entries added and known ones extended.
        """
        measurements = Measurements()
        measurements.add_measurement("known", [1.0])

        measurements.update_measurements({"known": [2.0], "new": [3.0]})

        assert measurements.data == {"known": [1.0, 2.0], "new": [3.0]}

    def test_update_measurements_merges_other_measurements(self):
        """
        Tests that another Measurements object is merged the same way
        a dictionary is.
        """
        measurements = Measurements()
        measurements.add_measurement("known", [1.0])
        other = Measurements()
        other.add_measurement("known", [2.0])
        other.add_measurement("new", [3.0])

        measurements.update_measurements(other)

        assert measurements.data == {"known": [1.0, 2.0], "new": [3.0]}

    def test_iadd_updates_measurements(self):
        """
        Tests that the += operator merges the measurements and returns
        the updated object.
        """
        measurements = Measurements()
        measurements.add_measurement("known", [1.0])

        measurements += {"known": [2.0]}

        assert measurements.data["known"] == [1.0, 2.0]

    def test_copy_returns_independent_dict(self):
        """
        Tests that copy returns a dictionary which is not affected by
        later additions of new measurement types.
        """
        measurements = Measurements()
        measurements.add_measurement("known", [1.0])

        copied = measurements.copy()
        measurements.add_measurement("new", [2.0])

        assert copied == {"known": [1.0]}

    def test_clear_removes_all_data(self):
        """
        Tests that clear removes all collected measurements.
        """
        measurements = Measurements()
        measurements.add_measurement("known", [1.0])

        measurements.clear()

        assert measurements.data == {}


class TestMeasurementsCollector:
    def test_save_measurements_creates_missing_directory(self, tmp_path: Path):
        """
        Tests that the directory for the measurements file is created
        when it does not exist.
        """
        resultpath = tmp_path / "missing" / "measurements.json"
        MeasurementsCollector.measurements += {"known": [1.0]}

        MeasurementsCollector.save_measurements(resultpath)

        assert json.loads(resultpath.read_text()) == {"known": [1.0]}

    def test_clear_removes_collected_measurements(self):
        """
        Tests that the collected measurements are dropped from the
        global collector.
        """
        MeasurementsCollector.measurements += {"known": [1.0]}

        MeasurementsCollector.clear()

        assert MeasurementsCollector.measurements.data == {}

    def test_set_unoptimized_nests_unoptimized_measurements(
        self, tmp_path: Path
    ):
        """
        Tests that the unoptimized measurements are stored under the
        UNOPTIMIZED key of the optimized ones.
        """
        optimized_path = tmp_path / "optimized.json"
        unoptimized_path = tmp_path / "unoptimized.json"
        optimized_path.write_text(json.dumps({"known": [1.0]}))
        unoptimized_path.write_text(json.dumps({"known": [2.0]}))

        MeasurementsCollector.set_unoptimized(optimized_path, unoptimized_path)

        assert json.loads(optimized_path.read_text()) == {
            "known": [1.0],
            Measurements.UNOPTIMIZED: {"known": [2.0]},
        }

    def test_set_unoptimized_removes_unoptimized_file(self, tmp_path: Path):
        """
        Tests that the unoptimized measurements file is deleted by
        default.
        """
        optimized_path = tmp_path / "optimized.json"
        unoptimized_path = tmp_path / "unoptimized.json"
        optimized_path.write_text(json.dumps({"known": [1.0]}))
        unoptimized_path.write_text(json.dumps({"known": [2.0]}))

        MeasurementsCollector.set_unoptimized(optimized_path, unoptimized_path)

        assert not unoptimized_path.exists()

    def test_set_unoptimized_keeps_unoptimized_file_when_requested(
        self, tmp_path: Path
    ):
        """
        Tests that the unoptimized measurements file is preserved when
        its removal is disabled.
        """
        optimized_path = tmp_path / "optimized.json"
        unoptimized_path = tmp_path / "unoptimized.json"
        optimized_path.write_text(json.dumps({"known": [1.0]}))
        unoptimized_path.write_text(json.dumps({"known": [2.0]}))

        MeasurementsCollector.set_unoptimized(
            optimized_path,
            unoptimized_path,
            remove_unoptimized_measurementsfile=False,
        )

        assert unoptimized_path.exists()


class TestTimeMeasurements:
    def test_duration_and_timestamp_are_collected(self):
        """
        Tests that the decorator collects the duration of the call and
        the timestamp taken after it.
        """

        @timemeasurements(
            "inference_step", get_time_func=fake_clock([1.0, 3.0, 4.0])
        )
        def measured():
            return None

        measured()

        assert MeasurementsCollector.measurements.data == {
            "inference_step": [2.0],
            "inference_step_timestamp": [4.0],
        }

    def test_consecutive_calls_are_appended(self):
        """
        Tests that every call of the wrapped function adds a new
        duration to the measurements.
        """

        @timemeasurements(
            "inference_step",
            get_time_func=fake_clock([1.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        )
        def measured():
            return None

        measured()
        measured()

        assert MeasurementsCollector.measurements.data == {
            "inference_step": [2.0, 1.0],
            "inference_step_timestamp": [4.0, 7.0],
        }

    def test_exception_is_propagated_without_collecting(self):
        """
        Tests that an exception raised by the wrapped function is
        propagated and no measurements are collected for the failed call.
        """

        @timemeasurements("inference_step", get_time_func=fake_clock([1.0]))
        def measured():
            raise ValueError("failure")

        with pytest.raises(ValueError, match="failure"):
            measured()

        assert MeasurementsCollector.measurements.data == {}


class TestTagMeasurements:
    def test_tag_is_collected(self):
        """
        Tests that the decorator collects the name of the tag along with
        the boundaries of the call.
        """

        @tagmeasurements("inference")
        def measured():
            return None

        measured()

        tags = MeasurementsCollector.measurements.data["tags"]
        assert len(tags) == 1
        assert tags[0]["name"] == "inference"
        assert tags[0]["start"] <= tags[0]["end"]

    def test_consecutive_calls_are_appended(self):
        """
        Tests that every call of the wrapped function adds a new tag to
        the measurements.
        """

        @tagmeasurements("inference")
        def measured():
            return None

        measured()
        measured()

        assert len(MeasurementsCollector.measurements.data["tags"]) == 2
