from unittest.mock import patch

import pytest
import tensorflow as tf

from kenning.converters.keras_converter import KerasConverter
from kenning.tests.core.conftest import DatasetModelRegistry


@pytest.fixture
def dummy_keras_model():
    _, model, _ = DatasetModelRegistry.get("keras")
    return {
        "path": model.get_path(),
        "io_spec": model.get_io_specification(),
    }


def test_to_keras(dummy_keras_model):
    conv = KerasConverter(source_model_path=dummy_keras_model["path"])
    model = conv.to_keras()
    assert isinstance(model, tf.keras.Model)


def test_to_tflite(dummy_keras_model):
    conv = KerasConverter(source_model_path=dummy_keras_model["path"])
    converter = conv.to_tflite()
    assert converter is not None
    assert hasattr(converter, "convert")

    tflite_bytes = converter.convert()
    assert isinstance(tflite_bytes, (bytes, bytearray))
    assert len(tflite_bytes) > 0


def test_to_onnx(dummy_keras_model):
    conv = KerasConverter(source_model_path=dummy_keras_model["path"])

    model = conv.to_onnx(
        dummy_keras_model["io_spec"]["processed_input"],
        dummy_keras_model["io_spec"]["output"],
    )

    from onnx import ModelProto

    assert model is not None
    assert len(model.graph.node) > 0
    assert model.graph.output[0].name == "out_layer"
    assert isinstance(model, ModelProto)


def test_to_tvm(dummy_keras_model):
    conv = KerasConverter(source_model_path=dummy_keras_model["path"])

    dummy_mod = "FAKE_MOD"
    dummy_params = {"w": "FAKE_PARAMS"}

    dummy_input_shape = (1, 1)
    dummy_input_type = "float32"

    with patch(
        "tvm.relay.frontend.from_keras", return_value=(dummy_mod, dummy_params)
    ) as mock_from_keras:
        mod, params = conv.to_tvm(
            input_shapes={"input": dummy_input_shape},
            dtypes={"input": dummy_input_type},
        )

        assert mod == dummy_mod
        assert params == dummy_params

        mock_from_keras.assert_called_once()


def test_invalid_model_path():
    conv = KerasConverter(source_model_path="nonexistent.h5")
    with pytest.raises(Exception):
        conv.to_keras()
