import pytest

from kenning.converters.tflite_converter import TFLiteConverter
from kenning.tests.core.conftest import DatasetModelRegistry


@pytest.fixture
def dummy_tflite_model():
    _, model, _ = DatasetModelRegistry.get("tflite")
    return {"path": model.get_path(), "io_spec": model.get_io_specification()}


def test_to_tvm(dummy_tflite_model):
    conv = TFLiteConverter(source_model_path=dummy_tflite_model["path"])

    mod, params = conv.to_tvm(
        io_spec=dummy_tflite_model["io_spec"],
    )

    import tvm

    assert isinstance(mod, tvm.ir.IRModule)
    assert isinstance(params, dict)


def test_to_onnx(dummy_tflite_model):
    conv = TFLiteConverter(source_model_path=dummy_tflite_model["path"])

    model = conv.to_onnx()

    from onnx import ModelProto

    assert model is not None
    assert isinstance(model, ModelProto)
