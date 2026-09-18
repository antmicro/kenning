from pathlib import Path

import pytest

from kenning.automl.auto_pytorch import AutoPyTorchML
from kenning.datasets.anomaly_detection_dataset import AnomalyDetectionDataset
from kenning.platforms.local import LocalPlatform
from kenning.tests.core.conftest import (
    get_dataset_random_mock,
)
from kenning.utils.automl_runner import AutoMLRunner


def create_spec(test_tmp_path: Path):
    conf = {
        "automl": {
            "type": "AutoPyTorchML",
        },
        "dataset": {
            "type": "AnomalyDetectionDataset",
            "parameters": {
                "dataset_root": f"{test_tmp_path}",
                "download_dataset": False,
                "n_best_models": 1,
            },
        },
        "optimizers": [
            {
                "type": "TFLiteCompiler",
                "parameters": {
                    "target": "default",
                    "compiled_model_path": f"{test_tmp_path}"
                    "automl-results/vae.tflite",
                    "inference_input_type": "float32",
                    "inference_output_type": "float32",
                },
            }
        ],
        "model_wrapper": {
            "type": "kenning.modelwrappers."
            "anomaly_detection.vae.PyTorchAnomalyDetectionVAE",
            "parameters": {
                "model_path": f"{test_tmp_path}/automl-results/13_12_10.0.pth",
            },
        },
    }
    return conf


@pytest.mark.xdist_group(name="automl")
def test_runner():
    dataset = get_dataset_random_mock(AnomalyDetectionDataset)
    test_tmp_path = dataset.root.resolve()
    conf = create_spec(test_tmp_path)

    model = AutoPyTorchML(
        dataset,
        LocalPlatform(),
        Path(test_tmp_path / "_autoPyTorch_tmp"),
        time_limit=1,
    )

    runner = AutoMLRunner(dataset, model, conf)
    try:
        next(runner.run("DEBUG"))
    except StopIteration:
        pass  # Time limit may be an issue here so it is not bad if there are
        # no items in the iterator
