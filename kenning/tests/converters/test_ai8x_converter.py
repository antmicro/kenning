import pytest
import torch

from kenning.converters.ai8x_converter import Ai8xConverter
from kenning.optimizers.ai8x import Ai8xTools
from kenning.tests.core.conftest import DatasetModelRegistry


@pytest.fixture
def dummy_ai8x_model():
    _, model, _ = DatasetModelRegistry.get("ai8x")
    return {
        "path": model.get_path(),
    }


def test_to_ai8x(dummy_ai8x_model, tmp_path):
    out_path = tmp_path / "ai8x_out.pt"

    converter = Ai8xConverter(source_model_path=dummy_ai8x_model["path"])

    converter.to_ai8x(out_path, Ai8xTools())

    saved = torch.load(out_path, weights_only=False)

    assert isinstance(saved, dict)
    assert saved["epoch"] == 0
    assert isinstance(saved["state_dict"], dict)
