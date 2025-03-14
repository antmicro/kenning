# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import csv
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Type, Union
from uuid import uuid4

import pytest
from tensorflow.keras.models import load_model as load_keras_model
from torch import save as torch_save

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.datasets.anomaly_detection_dataset import AnomalyDetectionDataset
from kenning.datasets.cnn_dailymail import CNNDailymailDataset
from kenning.datasets.coco_dataset import COCODataset2017
from kenning.datasets.imagenet_dataset import ImageNetDataset
from kenning.datasets.magic_wand_dataset import MagicWandDataset
from kenning.datasets.pet_dataset import PetDataset
from kenning.datasets.random_dataset import (
    RandomizedAnomalyDetectionDataset,
    RandomizedClassificationDataset,
    RandomizedDetectionSegmentationDataset,
    RandomizedTextDataset,
)
from kenning.datasets.visual_wake_words_dataset import VisualWakeWordsDataset
from kenning.modelwrappers.classification.tflite_magic_wand import (
    MagicWandModelWrapper,
)
from kenning.modelwrappers.object_detection.darknet_coco import (
    TVMDarknetCOCOYOLOV3,
)
from kenning.modelwrappers.object_detection.yolov4 import (
    ONNXYOLOV4,
)
from kenning.onnxconverters import onnx2torch
from kenning.optimizers.iree import IREECompiler
from kenning.optimizers.onnx import ONNXCompiler
from kenning.optimizers.tvm import TVMCompiler
from kenning.tests.conftest import get_tmp_path
from kenning.utils.resource_manager import PathOrURI, ResourceURI


def copy_model_to_tmp(model_path: PathOrURI) -> ResourceURI:
    """
    Copies model to tmp folder and returns its path.

    Parameters
    ----------
    model_path : PathOrURI
        Path or URI to the model file.

    Returns
    -------
    ResourceURI
        URI to the model copy.
    """
    tmp_path = get_tmp_path()
    if model_path.is_file():
        tmp_model_path = tmp_path.with_suffix(model_path.suffix)
        shutil.copy(model_path, tmp_model_path)

        json_path = model_path.with_suffix(model_path.suffix + ".json")
        if json_path.exists():
            shutil.copy(
                json_path,
                tmp_model_path.with_suffix(tmp_model_path.suffix + ".json"),
            )

        config_file = model_path.with_suffix(".cfg")
        if config_file.exists():
            shutil.copy(config_file, tmp_model_path.with_suffix(".cfg"))
    elif model_path.is_dir():
        tmp_model_path = tmp_path
        shutil.copytree(model_path, tmp_model_path)
    else:
        raise FileNotFoundError
    return ResourceURI(tmp_model_path)


def remove_file_or_dir(path: Union[Path, str]):
    """
    Removes directory of given path.

    Parameters
    ----------
    path : Union[Path, str]
        Path of given directory or file.
    """
    if Path(path).is_file():
        os.remove(path)
    elif Path(path).is_dir():
        shutil.rmtree(path)


class NonExistentAssetError(Exception):
    """
    Exception raised when a non-existent identifier is requested
    from DatasetModelRegistry.
    """

    pass


class DatasetModelRegistry:
    """
    Singleton containing a registry of pairs made of dataset mocks and
    temporary models.
    """

    debug: bool = False
    _registry: Dict[str, Tuple[Dataset, ModelWrapper]] = {}

    @classmethod
    def remove(cls, id: str):
        """
        Remove assets associated with the provided id.

        The method removes an associated pair of a mock dataset and
        model wrapper from the disk. Both of them are stored in a
        temporary location. But they may be removed earlier to reclaim
        some disk space. Not calling the method is not fatal;
        the temporary location is always purged at the end
        of life of a process.

        Parameters
        ----------
        id : str
            Unique identifier for a dataset and model.

        Raises
        ------
        NonExistentAssetError
            Raised if a method is called with `id` not present
            in the registry.

        """
        if DatasetModelRegistry.debug:
            return
        if id not in DatasetModelRegistry._registry:
            raise NonExistentAssetError(
                f"Cannot remove (dataset, model) pair with id = `{id}` "
                "because it is not present in the registry."
            )

        pair = DatasetModelRegistry._registry[id]
        if pair[0].root.exists():
            remove_file_or_dir(pair[0].root)
        if pair[1].get_path().exists():
            remove_file_or_dir(pair[1].get_path())
        del DatasetModelRegistry._registry[id]

    @classmethod
    def get(cls, framework: str) -> Tuple[Dataset, ModelWrapper, str]:
        """
        Returns a default model and dataset for a given framework.

        The returned dataset is a mock of a default dataset of the
        returned model. The third element of a tuple is id.
        Id is used to clear resources by call to `remove` method.

        Parameters
        ----------
        framework : str
            Name of framework.
        Parameters
        ----------
        framework : str
            Name of framework.

        Returns
        -------
        Tuple[Dataset, ModelWrapper, str]
            Tuple with: dataset, model for given framework,
            and id for the resources.
        """
        if framework == "keras":
            dataset = get_dataset_random_mock(MagicWandDataset)
            model_path = copy_model_to_tmp(
                ResourceURI(MagicWandModelWrapper.pretrained_model_uri)
            )
            model = MagicWandModelWrapper(model_path, dataset, from_file=True)

        elif framework == "tensorflow":
            dataset = get_dataset_random_mock(MagicWandDataset)
            model_path = get_tmp_path()
            keras_model = load_keras_model(
                ResourceURI(MagicWandModelWrapper.pretrained_model_uri),
                compile=False,
            )
            keras_model.save(model_path)
            model = MagicWandModelWrapper(model_path, dataset, from_file=True)
        elif framework == "tensorflow":
            dataset = get_dataset_random_mock(MagicWandDataset)
            model_path = get_tmp_path()
            keras_model = load_keras_model(
                ResourceURI(MagicWandModelWrapper.pretrained_model_uri),
                compile=False,
            )
            keras_model.save(model_path)
            model = MagicWandModelWrapper(model_path, dataset, from_file=True)

        elif framework == "tflite":
            dataset = get_dataset_random_mock(MagicWandDataset)
            model_path = copy_model_to_tmp(
                ResourceURI(
                    MagicWandModelWrapper.pretrained_model_uri
                ).with_suffix(".tflite")
            )
            model = MagicWandModelWrapper(model_path, dataset, from_file=True)
        elif framework == "tflite":
            dataset = get_dataset_random_mock(MagicWandDataset)
            model_path = copy_model_to_tmp(
                ResourceURI(
                    MagicWandModelWrapper.pretrained_model_uri
                ).with_suffix(".tflite")
            )
            model = MagicWandModelWrapper(model_path, dataset, from_file=True)

        elif framework == "onnx":
            dataset = get_dataset_random_mock(MagicWandDataset)
            model_path = get_tmp_path(suffix=".onnx")
            onnx_compiler = ONNXCompiler(dataset, model_path)
            onnx_compiler.init()
            onnx_compiler.compile(
                ResourceURI(MagicWandModelWrapper.pretrained_model_uri)
            )
            model = MagicWandModelWrapper(model_path, dataset, from_file=True)
        elif framework == "onnx":
            dataset = get_dataset_random_mock(MagicWandDataset)
            model_path = get_tmp_path(suffix=".onnx")
            onnx_compiler = ONNXCompiler(dataset, model_path)
            onnx_compiler.init()
            onnx_compiler.compile(
                ResourceURI(MagicWandModelWrapper.pretrained_model_uri)
            )
            model = MagicWandModelWrapper(model_path, dataset, from_file=True)

        elif framework == "torch":
            import dill
        elif framework == "torch":
            import dill

            dataset = get_dataset_random_mock(MagicWandDataset)
            onnx_model_path = get_tmp_path(suffix=".onnx")
            onnx_compiler = ONNXCompiler(dataset, onnx_model_path)
            onnx_compiler.init()
            onnx_compiler.compile(
                ResourceURI(MagicWandModelWrapper.pretrained_model_uri)
            )
            dataset = get_dataset_random_mock(MagicWandDataset)
            onnx_model_path = get_tmp_path(suffix=".onnx")
            onnx_compiler = ONNXCompiler(dataset, onnx_model_path)
            onnx_compiler.init()
            onnx_compiler.compile(
                ResourceURI(MagicWandModelWrapper.pretrained_model_uri)
            )

            torch_model = onnx2torch.convert(onnx_model_path)
            torch_model = onnx2torch.convert(onnx_model_path)

            model_path = get_tmp_path(suffix=".pth")
            torch_save(torch_model, model_path, pickle_module=dill)
            model_path = get_tmp_path(suffix=".pth")
            torch_save(torch_model, model_path, pickle_module=dill)

            model = MagicWandModelWrapper(model_path, dataset, from_file=True)
            model = MagicWandModelWrapper(model_path, dataset, from_file=True)

        elif framework == "darknet":
            dataset = get_dataset_random_mock(COCODataset2017)
            model_path = copy_model_to_tmp(
                ResourceURI(TVMDarknetCOCOYOLOV3.pretrained_model_uri)
            )
            model = TVMDarknetCOCOYOLOV3(model_path, dataset)
        elif framework == "darknet":
            dataset = get_dataset_random_mock(COCODataset2017)
            model_path = copy_model_to_tmp(
                ResourceURI(TVMDarknetCOCOYOLOV3.pretrained_model_uri)
            )
            model = TVMDarknetCOCOYOLOV3(model_path, dataset)

        elif framework == "iree":
            dataset = get_dataset_random_mock(MagicWandDataset)
            model_path = get_tmp_path(suffix=".vmfb")
            iree_compiler = IREECompiler(dataset, model_path)
            iree_compiler.init()
            iree_compiler.compile(
                ResourceURI(MagicWandModelWrapper.pretrained_model_uri)
            )
            model = MagicWandModelWrapper(model_path, dataset, from_file=True)
        elif framework == "iree":
            dataset = get_dataset_random_mock(MagicWandDataset)
            model_path = get_tmp_path(suffix=".vmfb")
            iree_compiler = IREECompiler(dataset, model_path)
            iree_compiler.init()
            iree_compiler.compile(
                ResourceURI(MagicWandModelWrapper.pretrained_model_uri)
            )
            model = MagicWandModelWrapper(model_path, dataset, from_file=True)

        elif framework == "tvm":
            dataset = get_dataset_random_mock(MagicWandDataset)
            model_path = get_tmp_path(suffix=".tar")
            tvm_compiler = TVMCompiler(
                dataset, model_path, model_framework="keras"
            )
            tvm_compiler.init()
            tvm_compiler.compile(
                ResourceURI(MagicWandModelWrapper.pretrained_model_uri)
            )
            model = MagicWandModelWrapper(model_path, dataset, from_file=True)
        elif framework in [
            "safetensors-native",
            "safetensors-awq",
            "safetensors-gptq",
        ]:
            raise UnknownFramework(
                f"LLM frameworks are not supported yet - {framework}"
            )
        else:
            raise UnknownFramework(f"Unknown framework: {framework}")
        elif framework == "tvm":
            dataset = get_dataset_random_mock(MagicWandDataset)
            model_path = get_tmp_path(suffix=".tar")
            tvm_compiler = TVMCompiler(
                dataset, model_path, model_framework="keras"
            )
            tvm_compiler.init()
            tvm_compiler.compile(
                ResourceURI(MagicWandModelWrapper.pretrained_model_uri)
            )
            model = MagicWandModelWrapper(model_path, dataset, from_file=True)
        elif framework in [
            "safetensors-native",
            "safetensors-awq",
            "safetensors-gptq",
        ]:
            raise UnknownFramework(
                f"LLM frameworks are not supported yet - {framework}"
            )
        else:
            raise UnknownFramework(f"Unknown framework: {framework}")

        model.save_io_specification(model.model_path)

        id = str(uuid4())
        DatasetModelRegistry._registry[id] = (dataset, model)
        return dataset, model, id


def get_dataset_download_path(dataset_cls: Type[Dataset]) -> Path:
    """
    Returns temporary download path for given dataset.

    Parameters
    ----------
    dataset_cls : Type[Dataset]
        Given dataset class.

    Returns
    -------
    Path
        Temporary path for dataset download.
    """
    return pytest.test_directory / "datasets" / dataset_cls.__name__


def get_reduced_dataset_path(dataset_cls: Type[Dataset]) -> Path:
    """
    Returns path to reduced dataset added to docker image.

    Parameters
    ----------
    dataset_cls : Type[Dataset]
        Given dataset class.

    Returns
    -------
    Path
        Path to reduced dataset.
    """
    return pytest.test_directory / dataset_cls.__name__


def get_dataset_random_mock(
    dataset_cls: Type[Dataset],
    modelwrapper_cls: Optional[Type[ModelWrapper]] = None,
) -> Dataset:
    """
    Return a mock for given dataset class.

    Parameters
    ----------
    dataset_cls : Type[Dataset]
        Dataset class to be mocked.

    Returns
    -------
    Dataset
        Mock of given dataset class.
    """
    if dataset_cls is PetDataset:
        return RandomizedClassificationDataset(
            get_tmp_path(),
            samplescount=37 * 5,
            numclasses=37,
            inputdims=(224, 224, 3),
        )
    if dataset_cls is ImageNetDataset:
        return RandomizedClassificationDataset(
            get_tmp_path(),
            samplescount=8 * 5,
            numclasses=8,
            inputdims=(224, 224, 3),
        )
    if dataset_cls is MagicWandDataset:
        dataset = RandomizedClassificationDataset(
            get_tmp_path(),
            samplescount=4 * 10,
            numclasses=4,
            inputdims=(128, 3, 1),
        )
        return dataset
    if dataset_cls is COCODataset2017:
        if modelwrapper_cls is ONNXYOLOV4:
            inputdims = (3, 608, 608)
        else:
            inputdims = (3, 416, 416)
        return RandomizedDetectionSegmentationDataset(
            get_tmp_path(),
            samplescount=8,
            numclasses=80,
            inputdims=inputdims,
        )
    if dataset_cls is VisualWakeWordsDataset:
        return RandomizedClassificationDataset(
            get_tmp_path(),
            samplescount=10,
            numclasses=2,
            inputdims=(480, 320, 3),
        )
    if dataset_cls is CNNDailymailDataset:
        return RandomizedTextDataset(
            get_tmp_path(),
        )
    if dataset_cls is AnomalyDetectionDataset:
        return RandomizedAnomalyDetectionDataset(
            get_tmp_path(),
            samplescount=16 * 16,
            numclasses=2,
            integer_classes=True,
            num_features=18,
            window_size=5,
        )
    raise NotImplementedError


@pytest.fixture(scope="module", autouse=True)
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
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_file = tmp_dir / "data.csv"
    with tmp_file.open("w") as fd:
        writer = csv.writer(fd)
        writer.writerows(data)

    # Specify csv_file param for AnomalyDetectionDataset
    default_init = AnomalyDetectionDataset.__init__
    AnomalyDetectionDataset.__init__ = lambda *args, **kwargs: default_init(
        *args, **kwargs, csv_file=str(tmp_file)
    )

    yield

    shutil.rmtree(tmp_dir)
    AnomalyDetectionDataset.__init__ = default_init


class UnknownFramework(ValueError):
    """
    Raised when unknown framework is passed.
    """

    pass
