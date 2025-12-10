import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from kenning.converters.torch_converter import TorchConverter
from kenning.core.exceptions import (
    CompilationError,
    ConversionError,
    ModelNotLoadedError,
)
from kenning.tests.core.conftest import DatasetModelRegistry


@pytest.fixture
def dummy_torch_model():
    _, model, _ = DatasetModelRegistry.get("torch")
    return {
        "path": model.get_path(),
        "io_spec": model.get_io_specification(),
    }


def test_to_torch_loads_full_model(dummy_torch_model):
    conv = TorchConverter(source_model_path=dummy_torch_model["path"])
    model = conv.to_torch()
    assert isinstance(model, torch.nn.Module)


def test_to_torch_raises_on_state_dict(tmp_path):
    path = tmp_path / "weights_only.pt"
    torch.save({"state_dict": {"a": 1}}, path)

    conv = TorchConverter(source_model_path=path)

    with pytest.raises(ModelNotLoadedError):
        conv.to_torch()


def test_to_onnx(dummy_torch_model):
    conv = TorchConverter(source_model_path=dummy_torch_model["path"])

    output_names = [
        spec["name"] for spec in dummy_torch_model["io_spec"]["output"]
    ]
    model = conv.to_onnx(
        dummy_torch_model["io_spec"]["processed_input"], output_names
    )

    assert model is not None
    assert len(model.graph.node) > 0
    assert model.graph.output[0].name == "out_layer"


def test_to_onnx_raises_on_invalid_model(tmp_path, dummy_torch_model):
    path = tmp_path / "invalid.pt"
    torch.save({"dummy": 123}, path)

    conv = TorchConverter(source_model_path=path)

    with pytest.raises(CompilationError):
        _ = conv.to_onnx(
            dummy_torch_model["io_spec"]["processed_input"],
            dummy_torch_model["io_spec"]["output"],
        )


class _DummyYamlWriter:
    def __init__(self):
        self.called = False

    def __call__(self, *args, **kwargs):
        self.called = True


class _DummyAi8xTools:
    ai8x_training_path = "/fake/path"

    def __init__(self):
        self.yamlwriter = _DummyYamlWriter()


class _DummyFused(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)


def test_to_ai8x(tmp_path):
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    model_path = tmp_path / "seq.pt"
    torch.save(model, model_path)

    (model_path.with_suffix(".pt.json")).write_text(
        json.dumps({"input": [{"shape": (1, 4)}]})
    )

    mock_ai8x_tools = _DummyAi8xTools()
    mock_fused_model = _DummyFused()

    with patch(
        "kenning.optimizers.ai8x_fuse.fuse_torch_sequential"
    ) as fuse_mock:
        fuse_mock.return_value = mock_fused_model

        conv = TorchConverter(source_model_path=model_path)
        out_path = tmp_path / "result.pt"

        conv.to_ai8x(out_path, mock_ai8x_tools, device_id=0)

        fuse_mock.assert_called_once()
        assert mock_ai8x_tools.yamlwriter.called


def test_to_ai8x_rejects_non_sequential(dummy_torch_model):
    conv = TorchConverter(source_model_path=dummy_torch_model["path"])
    with pytest.raises(ConversionError):
        conv.to_ai8x(Path("x"), MagicMock(), 0)


def test_to_tvm(dummy_torch_model):
    conv = TorchConverter(source_model_path=dummy_torch_model["path"])
    dummy_input_shape = tuple(
        dummy_torch_model["io_spec"]["processed_input"][0]["shape"]
    )
    dummy_input_type = "float32"
    mod, params = conv.to_tvm(
        input_shapes={"input": dummy_input_shape},
        dtypes={"input": dummy_input_type},
        conversion_func=None,
    )

    import tvm

    assert isinstance(mod, tvm.ir.IRModule)
    assert isinstance(params, dict)
