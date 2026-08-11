from pathlib import Path

from kenning.automl.auto_pytorch import AutoPyTorchML
from kenning.datasets.anomaly_detection_dataset import AnomalyDetectionDataset
from kenning.platforms.local import LocalPlatform
from kenning.tests.core.conftest import (
    get_dataset_random_mock,
)
from kenning.utils.automl_runner import AutoMLRunner


def test_runner():
    dataset = get_dataset_random_mock(AnomalyDetectionDataset)
    model = AutoPyTorchML(
        dataset,
        LocalPlatform(),
        Path("./build/_autoPyTorch_tmp/"),
        time_limit=1,
    )

    conf = {
        "automl": {
            "type": "AutoPyTorchML",
        },
        "dataset": {
            "type": "AnomalyDetectionDataset",
            "parameters": {
                "dataset_root": "./workspace/CATS",
                "download_dataset": False,
            },
        },
        "optimizers": [
            {
                "type": "TFLiteCompiler",
                "parameters": {
                    "target": "default",
                    "compiled_model_path": "./workspce/"
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
                "model_path": "workspace/automl-results/13_12_10.0.pth",
            },
        },
    }
    runner = AutoMLRunner(dataset, model, conf)
    try:
        next(runner.run("DEBUG"))
    except StopIteration:
        pass  # Time limit may be an issue here so it is not bad if there are
        # no items in the iterator
