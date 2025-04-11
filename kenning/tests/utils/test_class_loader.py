# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import abc
import argparse
import inspect
from contextlib import nullcontext as does_not_raise
from typing import Any, ContextManager, Dict, List, Set
from unittest.mock import patch

import pytest

import kenning.datasets.pet_dataset
from kenning.utils.class_loader import (
    ConfigKey,
    get_all_subclasses,
    get_base_classes_dict,
    get_command,
    get_kenning_submodule_from_path,
    load_class,
    merge_argparse_and_json,
    objs_from_argparse,
)


@pytest.mark.fast
class TestLoadClass:
    def test_empty_string(self):
        with pytest.raises(
            ModuleNotFoundError, match="None of the classes match"
        ):
            load_class("")

    def test_invalid_import_path(self):
        error_message = "No module named 'something"
        with pytest.raises(ModuleNotFoundError) as execinfo:
            load_class("something.that.is.not.implemented.here.YET")
        assert error_message in str(execinfo.value)

    def test_valid_import_path(self):
        loaded = load_class("kenning.datasets.pet_dataset.PetDataset")
        assert loaded is kenning.datasets.pet_dataset.PetDataset

    def test_capitalization(self):
        error_message = "No module named"
        with pytest.raises(ModuleNotFoundError) as execinfo:
            load_class("kenning.datasets.PET_DATASET.PetDataset")
        assert error_message in str(execinfo.value)

    def test_multiple_dots(self):
        error_message = "No module named 'kenning.datasets.pet_dataset.'"
        with pytest.raises(ModuleNotFoundError) as execinfo:
            load_class("kenning.datasets.pet_dataset.....PetDataset")
        assert error_message in str(execinfo.value)

    def test_class_name_only(self):
        loaded = load_class("PetDataset")
        assert loaded is kenning.datasets.pet_dataset.PetDataset

    def test_invalid_class_name(self):
        error_message = "None of the classes match"
        with pytest.raises(ModuleNotFoundError, match=error_message):
            load_class("NonExistentClass")


@pytest.mark.fast
class TestObjFrom:
    @staticmethod
    def json_cfg_simulated():
        return {
            "platform": {
                "type": "ZephyrPlatform",
                "parameters": {
                    "name": "max32690evkit/max32690/m4",
                    "simulated": True,
                    "zephyr_build_path": "./build/",
                    "uart_port": "/dev/ttyUSB0",
                },
            },
        }

    @staticmethod
    def json_cfg_not_simulated():
        return {
            "platform": {
                "type": "ZephyrPlatform",
                "parameters": {
                    "name": "max32690evkit/max32690/m4",
                    "simulated": False,
                    "zephyr_build_path": "./build/",
                    "uart_port": "/dev/ttyUSB0",
                },
            },
        }

    @pytest.mark.parametrize(
        "keys,json_cfg,expected_cfg,not_parsed,exec_expectation",
        [
            (
                set([ConfigKey.platform]),
                json_cfg_simulated(),
                json_cfg_not_simulated(),
                ["--no-simulated"],
                does_not_raise(),
            ),
            (
                set([ConfigKey.platform, ConfigKey.dataset]),
                json_cfg_not_simulated(),
                json_cfg_simulated(),
                ["--simulated"],
                does_not_raise(),
            ),
            (
                set(),
                json_cfg_simulated(),
                json_cfg_simulated(),
                ["--no-simulated"],
                pytest.raises(argparse.ArgumentError),
            ),
            (
                set([ConfigKey.platform]),
                json_cfg_simulated(),
                json_cfg_simulated(),
                [],
                does_not_raise(),
            ),
        ],
    )
    def test_merge_argparse_and_json(
        self,
        keys: Set[ConfigKey],
        json_cfg: Dict[str, Any],
        expected_cfg: Dict[str, Any],
        not_parsed: List[str],
        exec_expectation: ContextManager,
    ):
        with patch("kenning.utils.class_loader.get_command") as mock:
            mock.return_value = "kenning"
            with exec_expectation:
                merge_argparse_and_json(
                    keys, json_cfg, argparse.Namespace(help=None), not_parsed
                )
                assert json_cfg == expected_cfg

    def test_objs_from_argparse_callback(self):
        def required(_):
            raise RuntimeError

        with pytest.raises(RuntimeError):
            objs_from_argparse(
                args=argparse.Namespace(),
                not_parsed=[],
                keys=set(),
                required=required,
            )


@pytest.mark.fast
class TestGetKenningSubmoduleFromPath:
    def test_multiple_kenning(self):
        x = "/tmp/smth/anything/kenning/kenning/kenning/core/"
        y = "kenning.core"
        assert get_kenning_submodule_from_path(x) == y

    def test_not_valid(self):
        with pytest.raises(ValueError) as execinfo:
            x = "/tmp/smth/anything/"
            get_kenning_submodule_from_path(x)
        assert "not in tuple" in str(execinfo.value)

    def test_valid_class_loader_path(self):
        x = "/usr/lib/python3.10/site-packages/kenning/utils/class_loader.py"
        y = "kenning.utils.class_loader"
        assert get_kenning_submodule_from_path(x) == y

    def test_module_name(self):
        x = "/usr/lib/python3.10/site-packages/kenning/"
        y = "kenning"
        assert get_kenning_submodule_from_path(x) == y

    def test_invalid_capitalization(self):
        with pytest.raises(ValueError) as execinfo:
            x = "/usr/lib/python3.10/site-packages/Kenning/utils/class_loader.py"  # noqa: E501
            get_kenning_submodule_from_path(x)
        assert "not in tuple" in str(execinfo.value)


@pytest.mark.fast
class TestGetCommand:
    def test_invalid_capitalization(self):
        with pytest.raises(ValueError) as execinfo:
            x = [
                "/usr/lib/python3.10/Kenning/utils/class_loader.py",
            ]
            get_command(x)
        assert "not in tuple" in str(execinfo.value)

    def test_different_flags(self):
        x = [
            "/usr/lib/python3.10/kenning/scenarios/model_training.py",
            "--help",
            "-h",
        ]
        y = [
            "python -m kenning.scenarios.model_training \\",
            "    --help \\",
            "    -h",
        ]
        assert get_command(x) == y

    def test_only_file(self):
        x = [
            "/usr/lib/python3.10/kenning/scenarios/model_training.py",
        ]
        y = [
            "python -m kenning.scenarios.model_training",
        ]
        assert get_command(x) == y

    def test_multiple_flags(self):
        x = [
            "/usr/lib/python3.10/kenning/scenarios/model_training.py",
            "--modelwrapper-cls kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2",  # noqa: E501
            "--dataset-cls kenning.datasets.pet_dataset.PetDataset",
            "--logdir",
            "build/logs",
            "--dataset-root",
            "build/pet-dataset",
            "--model-path",
            "build/trained-model.h5",
            "--batch-size",
            "32",
            "--learning-rate",
            "0.0001",
            "--num-epochs",
            "50",
        ]
        y = [
            "python -m kenning.scenarios.model_training \\",
            "    --modelwrapper-cls kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \\",  # noqa: E501
            "    --dataset-cls kenning.datasets.pet_dataset.PetDataset \\",
            "    --logdir \\",
            "        build/logs \\",
            "    --dataset-root \\",
            "        build/pet-dataset \\",
            "    --model-path \\",
            "        build/trained-model.h5 \\",
            "    --batch-size \\",
            "        32 \\",
            "    --learning-rate \\",
            "        0.0001 \\",
            "    --num-epochs \\",
            "        50",
        ]
        assert get_command(x) == y

    def test_empty_command(self):
        with pytest.raises(IndexError) as execinfo:
            x = []
            get_command(x)
        assert "list index out of range" in str(execinfo.value)

    def test_empty_flags(self):
        x = [
            "/usr/lib/python3.10/kenning/scenarios/model_training.py",
            "",
        ]
        y = ["python -m kenning.scenarios.model_training"]
        assert get_command(x) == y


class TestGetAllSubclasses:
    @pytest.mark.parametrize(
        "module_path,cls",
        [
            pytest.param(
                module_path,
                cls,
                marks=(
                    pytest.mark.skip() if module == "onnxconversions" else ()
                ),
            )
            for module, (module_path, cls) in get_base_classes_dict().items()
        ],
    )
    def test_get_all_subclasses_should_return_non_abstract_classes(
        self,
        module_path: str,
        cls: type,
    ):
        """
        Tests loading all subclasses of given class.
        """
        subclasses = get_all_subclasses(
            module_path, cls, raise_exception=False
        )

        for subcls in subclasses:
            assert isinstance(subcls, type)
            assert not inspect.isabstract(subcls)
            assert abc.ABC not in subcls.__bases__

    @pytest.mark.parametrize(
        "module_path,cls", get_base_classes_dict().values()
    )
    def test_get_all_subclasses_should_not_load_classes_if_specified(
        self,
        module_path: str,
        cls: type,
    ):
        """
        Tests retrieving without loading all subclasses of given class.
        """
        subclasses = get_all_subclasses(
            module_path, cls, raise_exception=False, import_classes=False
        )

        for subcls, module in subclasses:
            assert isinstance(subcls, str)
            assert isinstance(module, str)
