# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from kenning.core.drawing import (
    Barplot,
    BubblePlot,
    ConfusionMatrixPlot,
    LinePlot,
    RadarChart,
    RecallPrecisionCurvesPlot,
    RecallPrecisionGradients,
    TruePositiveIoUHistogram,
    TruePositivesPerIoURangeHistogram,
    ViolinComparisonPlot,
)
from kenning.tests.conftest import get_tmp_path


def get_colors(n_colors):
    cmap = plt.get_cmap("nipy_spectral")
    return [cmap(i) for i in np.linspace(0.0, 1.0, n_colors)]


def check_output_plot(func):
    @pytest.mark.parametrize("backend", ["matplotlib", "bokeh"])
    def wrapper(backend, *args, **kwargs):
        format = ""
        if backend == "matplotlib":
            format = "png"
        else:
            format = "html"
        temp_file = get_tmp_path()
        plot = func(*args, **kwargs)
        if plot:
            assert (
                plot.plot(
                    output_path=temp_file,
                    output_formats=[format],
                    backend=backend,
                )
                is None
            )
            assert temp_file.with_suffix("." + format).exists()

    return wrapper


@check_output_plot
def test_basic_lineplot():
    line1 = np.arange(0, 10, 1)
    line2 = np.arange(2, 0, -0.2)
    plot = LinePlot(
        title=("title"),
        x_label="x",
        y_label="y",
        lines=[(line1, line2)],
        colors=get_colors(1),
        color_offset=0,
    )
    return plot


@check_output_plot
def test_basic_labeled_lineplot():
    line1 = np.arange(0, 10, 1)
    line2 = np.arange(2, 0, -0.2)
    plot = LinePlot(
        title=("title"),
        x_label="x",
        y_label="y",
        lines=[(line1, line2)],
        lines_labels=["line1", "line2"],
        colors=get_colors(1),
        color_offset=0,
    )

    return plot


@check_output_plot
def test_vertical_lineplot():
    line1 = np.array([0, 0, 0, 0, 0])
    line2 = np.arange(0, 1, 0.2)
    plot = LinePlot(
        title=("title"),
        x_label="x",
        y_label="y",
        lines=[(line1, line2)],
        colors=get_colors(1),
        color_offset=0,
    )

    return plot


@check_output_plot
def test_horizontal_lineplot():
    line1 = np.arange(0, 1, 0.2)
    line2 = np.array([0, 0, 0, 0, 0])
    plot = LinePlot(
        title=("title"),
        x_label="x",
        y_label="y",
        lines=[(line1, line2)],
        colors=get_colors(1),
        color_offset=0,
    )

    return plot


@check_output_plot
def test_empty_lineplot():
    line1 = np.array([])
    line2 = np.array([])
    plot = LinePlot(
        title=("title"),
        x_label="x",
        y_label="y",
        lines=[(line1, line2)],
        colors=get_colors(1),
        color_offset=0,
    )

    return plot


@check_output_plot
def test_barplot():
    x_series = ["foo", "bar", "baz"]
    y_series = {
        "foo": [1, 2, 3],
        "bar": [4, 5, 6],
        "baz": [7, 8, 9],
    }
    plot = Barplot(x_series, y_series, "x_label", "y_label")
    return plot


@check_output_plot
def test_zeroed_barplot():
    x_series = ["foo", "bar", "baz"]
    y_series = {
        "foo": [0, 0, 0],
        "bar": [0, 0, 0],
        "baz": [0, 0, 0],
    }
    plot = Barplot(x_series, y_series, "x_label", "y_label")
    return plot


@patch("kenning.core.drawing.plt.show")
def test_empty_barplot(mock_show):
    x_series = []
    y_series = {}

    with pytest.raises(ValueError) as e:
        Barplot(x_series, y_series, "x_label", "y_label")

    assert "Barplot rendering error:" in str(e.value)


@check_output_plot
def test_bubbleplot():
    x_data = [1.0, 2.0, 3.0, 4.0]
    x_label = "x"
    y_data = [1.0, 2.0, 3.0, 4.0]
    y_label = "y"
    size_data = [2, 3, 0.5, -1]
    size_label = "size"
    bubble_labels = ["foo", "bar", "baz", "ban"]
    plot = BubblePlot(
        x_data, x_label, y_data, y_label, size_data, size_label, bubble_labels
    )
    return plot


@check_output_plot
def test_zeroed_bubbleplot():
    x_data = [0, 0, 0, 0]
    x_label = "x"
    y_data = [0, 0, 0, 0]
    y_label = "y"
    size_data = [0, 0, 0, 0]
    size_label = "size"
    bubble_labels = ["foo", "bar", "baz", "ban"]
    plot = BubblePlot(
        x_data, x_label, y_data, y_label, size_data, size_label, bubble_labels
    )
    return plot


@patch("kenning.core.drawing.plt.show")
def test_empty_bubbleplot(mock_show):
    x_data = []
    x_label = "x"
    y_data = []
    y_label = "y"
    size_data = []
    size_label = "size"
    bubble_labels = ["foo", "bar", "baz", "ban"]
    with pytest.raises(ValueError) as e:
        BubblePlot(
            x_data,
            x_label,
            y_data,
            y_label,
            size_data,
            size_label,
            bubble_labels,
        )

    assert "BubblePlot rendering error" in str(e.value)


@check_output_plot
def test_confusionmatrixplot():
    confusion_matrix = [[1, 0, 0], [0, 0.8, 0.2], [0, 0, 1]]
    class_names = ["ham", "jam", "spam"]
    plot = ConfusionMatrixPlot(confusion_matrix, class_names)
    return plot


@check_output_plot
def test_big_confusionmatrixplot():
    num_classes = 100
    confusion_matrix = [[1] * num_classes] * num_classes
    class_names = [f"class no. {i}" for i in range(num_classes)]
    plot = ConfusionMatrixPlot(confusion_matrix, class_names)
    return plot


@check_output_plot
def test_zeroed_confusionmatrixplot():
    confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    class_names = ["ham", "jam", "spam"]
    plot = ConfusionMatrixPlot(confusion_matrix, class_names)
    return plot


@check_output_plot
def test_empty_confusionmatrixplot():
    confusion_matrix = [[]]
    class_names = []
    with pytest.raises(ValueError) as e:
        ConfusionMatrixPlot(confusion_matrix, class_names)

    assert "Confusion matrix rendering error:" in str(e.value)


@check_output_plot
def test_invalid_confusionmatrixplot():
    confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0]]
    class_names = ["ham", "jam", "incorrect"]
    with pytest.raises(ValueError) as e:
        ConfusionMatrixPlot(confusion_matrix, class_names)

    assert "Confusion matrix rendering error:" in str(e.value)


@check_output_plot
def test_radarchart():
    labels = ["foo", "bar", "baz"]
    metrics = {
        "foo": [1, 2, 3],
        "bar": [4, 5, 6],
        "baz": [7, 8, 9],
    }
    plot = RadarChart(metrics, labels)
    return plot


@check_output_plot
def test_empty_radarchart():
    labels = []
    metrics = {}
    with pytest.raises(ValueError) as e:
        RadarChart(metrics, labels)
    assert "RadarChart rendering error:" in str(e.value)


@check_output_plot
def test_zeroed_radarchart():
    labels = ["foo", "bar"]
    metrics = {"foo": [0, 0], "bar": [0, 0]}
    plot = RadarChart(metrics, labels)

    return plot


@check_output_plot
def test_recallprecisioncurvesplot():
    lines = [
        ([0, 0.5, 1], [0, 0.5, 1]),
        ([1, 0.6, 0.7], [0, 0.5, 1]),
        ([0.5, 1, 0], [0, 0.5, 1]),
    ]

    plot = RecallPrecisionCurvesPlot(
        lines=lines,
        class_names=["foo", "bar", "baz"],
    )
    return plot


@check_output_plot
def test_zeroed_recallprecisioncurvesplot():
    lines = [
        ([0, 0, 0], [0, 0, 0]),
        ([0, 0, 0], [0, 0, 0]),
        ([0, 0, 0], [0, 0, 0]),
    ]

    plot = RecallPrecisionCurvesPlot(
        lines=lines,
        class_names=["foo", "bar", "baz"],
    )
    return plot


@check_output_plot
def test_empty_recallprecisioncurvesplot():
    lines = []

    plot = RecallPrecisionCurvesPlot(
        lines=lines,
        class_names=[],
    )
    return plot


@check_output_plot
def test_recallprecisiongradients():
    lines = [
        ([0, 0.5, 1], [1, 0.5, 1]),
        ([1, 0.6, 0.7], [0.6, 0.5, 1]),
        ([0.5, 1, 0], [0.8, 0.5, 1]),
    ]

    class_names = (["foo", "bar", "baz"],)

    avg_precisions = [sum(line[1]) / len(line[1]) for line in lines]
    map = sum(avg_precisions) / len(avg_precisions)

    plot = RecallPrecisionGradients(
        lines=lines,
        class_names=class_names,
        avg_precisions=avg_precisions,
        mean_avg_precision=map,
    )

    return plot


@check_output_plot
def test_zeroed_recallprecisiongradients():
    lines = [
        ([0, 0, 0], [0, 0, 0]),
        ([0, 0, 0], [0, 0, 0]),
        ([0, 0, 0], [0, 0, 0]),
    ]

    class_names = (["foo", "bar", "baz"],)

    avg_precisions = [sum(line[1]) / len(line[1]) for line in lines]
    map = sum(avg_precisions) / len(avg_precisions)

    plot = RecallPrecisionGradients(
        lines=lines,
        class_names=class_names,
        avg_precisions=avg_precisions,
        mean_avg_precision=map,
    )

    return plot


@check_output_plot
def test_empty_recallprecisiongradients():
    lines = []
    class_names = ([],)
    avg_precisions = []
    map = 0

    plot = RecallPrecisionGradients(
        lines=lines,
        class_names=class_names,
        avg_precisions=avg_precisions,
        mean_avg_precision=map,
    )

    return plot


@check_output_plot
def test_truepositiveiouhistogram():
    plot = TruePositiveIoUHistogram([0.25, 0.6, 1], ["foo", "bar", "baz"])
    return plot


@check_output_plot
def test_zeroed_truepositiveiouhistogram():
    plot = TruePositiveIoUHistogram([0, 0, 0], ["foo", "bar", "baz"])
    return plot


@check_output_plot
def test_empty_truepositiveiouhistogram():
    plot = TruePositiveIoUHistogram([], [])
    return plot


@check_output_plot
def test_truepositivesperiourangehistogram():
    plot = TruePositivesPerIoURangeHistogram(
        [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    )
    return plot


@check_output_plot
def test_zeroed_truepositivesperiourangehistogram():
    plot = TruePositivesPerIoURangeHistogram(
        [0, 0, 0, 0],
    )
    return plot


@check_output_plot
def test_empty_truepositivesperiourangehistogram():
    plot = TruePositivesPerIoURangeHistogram([])
    return plot


@check_output_plot
def test_violincomparisonplot():
    data = {
        "foo": [[1, 2], [3, 4]],
        "bar": [[2, 1], [4, 3]],
        "baz": [[1, 2], [0, 4]],
    }
    labels = ["meas1", "meas2"]
    plot = ViolinComparisonPlot(data, labels)
    return plot


@check_output_plot
def test_zeroed_violincomparisonplot():
    data = {
        "foo": [[0, 0], [0, 0]],
        "bar": [[0, 0], [0, 0]],
        "baz": [[0, 0], [0, 0]],
    }
    labels = ["meas1", "meas2"]
    plot = ViolinComparisonPlot(data, labels)
    return plot


@check_output_plot
def test_empty_violincomparisonplot():
    data = {"foo": [[], []], "bar": [[], []], "baz": [[], []]}
    labels = ["meas1", "meas2"]
    with pytest.raises(ValueError) as e:
        ViolinComparisonPlot(data, labels)
    assert "Violin Plot rendering error:" in str(e.value)


@check_output_plot
def test_invalid_violincomparisonplot():
    data = {
        "foo": [[0, 0], [0, 0]],
        "bar": [[0, 0], [0, 0]],
        "baz": [[0, 0], [0, 0]],
    }
    labels = ["foo", "bar", "incorrect"]
    with pytest.raises(ValueError) as e:
        ViolinComparisonPlot(data, labels)
    assert "Violin Plot rendering error:" in str(e.value)


@pytest.mark.parametrize("backend", ["matplotlib", "bokeh"])
@pytest.mark.parametrize("format", ["html", "png", "svg"])
def test_export(backend, format):
    if backend == "bokeh" and format != "html":
        pytest.skip(reason="requires Selenium and geckodriver/ChromeDrive")
    line1 = np.arange(0, 10, 1)
    line2 = np.arange(2, 0, -0.2)
    plot = LinePlot(
        title=("title"),
        x_label="x",
        y_label="y",
        lines=[(line1, line2)],
        colors=get_colors(1),
        color_offset=0,
    )
    path = get_tmp_path()
    formats = [format]
    assert (
        plot.plot(backend=backend, output_formats=formats, output_path=path)
        is None
    )
    assert path.with_suffix("." + format).exists()
