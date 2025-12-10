import pytest

from kenning.converters.tflite_converter import TFLiteConverter
from kenning.tests.core.conftest import DatasetModelRegistry


@pytest.fixture
def dummy_tflite_model():
    _, model, _ = DatasetModelRegistry.get("tflite")
    return {
        "path": model.get_path(),
    }


def test_to_tvm(dummy_tflite_model):
    conv = TFLiteConverter(source_model_path=dummy_tflite_model["path"])
    dummy_input_shape = (1, 1)
    dummy_input_type = "float32"

    mod, params = conv.to_tvm(
        input_shapes={"input": dummy_input_shape},
        dtypes={"input": dummy_input_type},
    )

    import tvm

    assert isinstance(mod, tvm.ir.IRModule)
    assert isinstance(params, dict)


def test_to_onnx(dummy_tflite_model):
    conv = TFLiteConverter(source_model_path=dummy_tflite_model["path"])

    model = conv.to_onnx([], [])

    from onnx import ModelProto

    assert model is not None
    assert isinstance(model, ModelProto)
