# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import random
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pytest
from pytest_mock import MockerFixture

from kenning.cli.command_template import (
    AUTOML,
    OPTIMIZE,
    REPORT,
    TEST,
    ParserHelpException,
)
from kenning.cli.parser import USED_SUBCOMMANDS
from kenning.datasets.anomaly_detection_dataset import AnomalyDetectionDataset
from kenning.scenarios.automl import AutoMLCommand

ConfigType = Iterable[Tuple[Path, Dict]]


@pytest.fixture()
def define_anomaly_detection_csv_file():
    """
    Creates random CSV file for AnomalyDetectionDataset
    and overrides init.
    """
    # Generate random data

    columns = 10
    data = [["a"] * columns]
    for _ in range(1000):
        data.append([random.random() for _ in range(columns)])

    # Save data to tmp file
    dataset_dir = Path("./workspace/CATS/")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    csv_file = dataset_dir / "data"
    with csv_file.open("w+") as fd:
        writer = csv.writer(fd)
        writer.writerows(data)

    # Specify csv_file param for AnomalyDetectionDataset
    default_anomaly_init = AnomalyDetectionDataset.__init__

    def mock_anomaly_init(*args, **kwargs):
        kwargs["csv_file"] = str(csv_file)
        return default_anomaly_init(*args, **kwargs)

    AnomalyDetectionDataset.__init__ = mock_anomaly_init
    yield


@pytest.fixture
def automl_runner_mock(mocker: MockerFixture, automl_conf):
    module = "kenning.scenarios.automl"
    name = f"{module}.AutoMLRunner"
    mock = mocker.Mock()

    def run(output, *args, **kwargs):
        return automl_conf

    mocker.patch(name, return_value=mock)
    mocker.patch(f"{name}.from_objs_dict", return_value=mock)
    mock.run = run
    Path("./workspace/automl-results").mkdir(exist_ok=True, parents=True)
    mock.autoML.output_directory = Path("./workspace/automl-results")
    mock.autoML.n_best_models = 1

    name = f"{module}.get_command"
    mocker.patch(name, return_value="")


@pytest.fixture
def create_spec(define_anomaly_detection_csv_file):
    Path("./workspace/CATS").mkdir(parents=True, exist_ok=True)
    spec = {
        "automl": {
            "type": "AutoPyTorchML",
            "parameters": {
                "time_limit": 1,
                "seed": 13,
                "use_models": ["PyTorchAnomalyDetectionVAE"],
                "n_best_models": 5,
                "output_directory": "./workspace/automl-results",
            },
        },
        "platform": {
            "type": "LocalPlatform",
        },
        "dataset": {
            "type": "AnomalyDetectionDataset",
            "parameters": {
                "dataset_root": "./workspace/CATS",
                "csv_file": "./workspace/CATS/data",
                "split_fraction_test": 0.1,
                "split_seed": 12,
                "inference_batch_size": 1,
                "download_dataset": False,
            },
        },
        "optimizers": [
            {
                "type": "TFLiteCompiler",
                "parameters": {
                    "target": "default",
                    "compiled_model_path": "./workspace"
                    "/automl-results/vae.tflite",
                    "inference_input_type": "float32",
                    "inference_output_type": "float32",
                },
            }
        ],
    }
    return json.dumps(spec)


@pytest.fixture
def automl_conf(define_anomaly_detection_csv_file):
    path = Path("./workspace/test.yml")
    Path("./workspace/CATS").mkdir(parents=True, exist_ok=True)

    conf = {
        "automl": {
            "type": "AutoPyTorchML",
            "platform": "LocalPlatform",
        },
        "dataset": {
            "type": "AnomalyDetectionDataset",
            "parameters": {
                "dataset_root": "./workspace/CATS",
                "csv_file": "./workspace/CATS/data",
                "download_dataset": False,
            },
        },
        "model_wrapper": {
            "type": "kenning.modelwrappers."
            "anomaly_detection.vae.PyTorchAnomalyDetectionVAE",
            "parameters": {
                "model_path": "workspace/automl-results/13_12_10.0.pth",
            },
        },
    }
    return [(path, conf)]


def test_help(create_spec):
    jfile = create_spec
    Path("./workspace").mkdir(exist_ok=True)

    with open("./workspace/config.json", "w+") as js:
        js.write(jfile)

    mock_args = Namespace(
        **{USED_SUBCOMMANDS: [AUTOML, OPTIMIZE, TEST, REPORT]},
        help=True,
        json_cfg="./workspace/config.json",
        verbosity="INFO",
        use_previous_results=False,
        allow_failures=False,
    )
    with pytest.raises(ParserHelpException):
        AutoMLCommand.run(mock_args)


def test_configure_parser():
    AutoMLCommand.configure_parser()


@pytest.mark.dependency()
def test_automl_scenario_run_cfg(create_spec, automl_runner_mock):
    jfile = create_spec
    Path("./workspace").mkdir(exist_ok=True)

    with open("./workspace/config.json", "w+") as js:
        js.write(jfile)

    mock_args = Namespace(
        **{USED_SUBCOMMANDS: [AUTOML, OPTIMIZE, TEST, REPORT]},
        help=False,
        json_cfg="./workspace/config.json",
        verbosity="INFO",
        use_previous_results=False,
        allow_failures=False,
    )
    AutoMLCommand.run(mock_args)


@pytest.mark.dependency(depends=["test_automl_scenario_run_cfg"])
def test_with_previous_results(create_spec, automl_runner_mock):
    jfile = create_spec
    Path("./workspace").mkdir(exist_ok=True)

    with open("./workspace/config.json", "w+") as js:
        js.write(jfile)

    mock_args = Namespace(
        **{USED_SUBCOMMANDS: [AUTOML, OPTIMIZE, TEST]},
        help=False,
        json_cfg="./workspace/config.json",
        verbosity="INFO",
        use_previous_results=True,
        allow_failures=False,
    )
    AutoMLCommand.run(mock_args)
