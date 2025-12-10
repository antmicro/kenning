import pytest
from onnx import ModelProto

from kenning.converters.onnx_converter import OnnxConverter
from kenning.tests.core.conftest import DatasetModelRegistry


@pytest.fixture
def dummy_onnx_model():
    _, model, _ = DatasetModelRegistry.get("onnx")
    return {
        "path": model.get_path(),
    }


def test_to_onnx(dummy_onnx_model):
    conv = OnnxConverter(source_model_path=dummy_onnx_model["path"])
    model = conv.to_onnx([], [])
    assert isinstance(model, ModelProto)
    assert model is not None
    assert len(model.graph.node) > 0


def test_to_torch(dummy_onnx_model):
    conv = OnnxConverter(source_model_path=dummy_onnx_model["path"])
    model = conv.to_torch()
    import torch

    assert isinstance(model, torch.nn.Module)
