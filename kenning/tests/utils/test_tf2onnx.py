# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import functools
from pathlib import Path
from tempfile import NamedTemporaryFile

import keras
import numpy as np
import onnxruntime
import pytest
import scipy.stats
import tf2onnx

import kenning.utils.tf2onnx  # noqa: F401
from kenning.converters.keras_converter import KerasConverter

INPUT_SIZE = 10
TOLERANCE = 0.0001


def gelu(x: np.ndarray, approximate: bool) -> np.ndarray:
    if approximate:
        return (
            0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        )

    return x * scipy.stats.norm.cdf(x)


def create_gelu_model(approximate: bool):
    activation = functools.partial(
        keras.activations.gelu, approximate=approximate
    )

    model = keras.Sequential(
        [keras.layers.Activation(activation, input_shape=(INPUT_SIZE,))]
    )

    with (
        NamedTemporaryFile() as tflite_temp,
        NamedTemporaryFile() as onnx_temp,
    ):
        tflite_model = KerasConverter(Path()).to_tflite(model).convert()

        tflite_temp.write(tflite_model)
        tflite_temp.flush()

        tf2onnx.convert.from_tflite(
            tflite_temp.name, output_path=onnx_temp.name
        )

        return onnxruntime.InferenceSession(onnx_temp.name)


@pytest.fixture(scope="module")
def model_variants():
    return (
        create_gelu_model(approximate=False),
        create_gelu_model(approximate=True),
    )


@pytest.fixture(scope="module")
def gelu_variants():
    return (
        lambda x: gelu(x, approximate=False),
        lambda x: gelu(x, approximate=True),
    )


def check_predictions(
    session: onnxruntime.InferenceSession,
    input_data: np.ndarray,
    expected: np.ndarray,
):
    input_data = input_data.astype(np.float32)

    input_name = session.get_inputs()[0].name
    actual = session.run(None, {input_name: input_data})

    return np.allclose(expected, actual, atol=TOLERANCE)


def test_random(model_variants, gelu_variants):
    for model, gelu in zip(model_variants, gelu_variants):
        input_data = np.random.random_sample((1, INPUT_SIZE))
        expected = gelu(input_data)

        assert check_predictions(model, input_data, expected)


def test_manual(model_variants, gelu_variants):
    for model, gelu in zip(model_variants, gelu_variants):
        input_data = np.zeros((1, INPUT_SIZE))
        expected = gelu(input_data)

        assert check_predictions(model, input_data, expected)

        input_data = np.asarray([[2] * INPUT_SIZE])
        expected = gelu(input_data)

        assert check_predictions(model, input_data, expected)

        input_data = np.asarray([[-3.33] * INPUT_SIZE])
        expected = gelu(input_data)

        assert check_predictions(model, input_data, expected)
