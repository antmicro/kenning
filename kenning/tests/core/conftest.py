# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from typing import Tuple, Type
from pathlib import Path
import tempfile
import shutil
import os
from tensorflow.keras.models import load_model

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.datasets.pet_dataset import PetDataset
from kenning.datasets.imagenet_dataset import ImageNetDataset
from kenning.datasets.coco_dataset import COCODataset2017
from kenning.datasets.magic_wand_dataset import MagicWandDataset
from kenning.datasets.random_dataset import RandomizedClassificationDataset
from kenning.datasets.random_dataset import RandomizedDetectionSegmentationDataset  # noqa: 501
from kenning.modelwrappers.classification.tflite_magic_wand import MagicWandModelWrapper    # noqa: 501
from kenning.modelwrappers.classification.pytorch_pet_dataset import PyTorchPetDatasetMobileNetV2   # noqa: 501
from kenning.modelwrappers.classification.tensorflow_pet_dataset import TensorFlowPetDatasetMobileNetV2    # noqa: 501
from kenning.modelwrappers.detectors.yolov4 import ONNXYOLOV4
from kenning.modelwrappers.detectors.darknet_coco import TVMDarknetCOCOYOLOV3


KENNING_MODELS_PATH = Path(r'kenning/resources/models/')


def get_tmp_path() -> Path:
    """
    Generates temporary path

    Returns
    -------
    Path :
        Temporary path
    """
    return (pytest.test_directory / 'tmp' /
            next(tempfile._get_candidate_names()))


def copy_model_to_tmp(modelpath: Path) -> Path:
    """
    Copies model to tmp folder and returns its path.

    Parameters
    ----------
    modelpath : Path
        Path to the model

    Returns
    -------
    Path :
        Path to the model copy
    """
    tmp_path = get_tmp_path()
    if modelpath.is_file():
        tmp_modelpath = tmp_path.with_suffix(modelpath.suffix)
        shutil.copy(modelpath, tmp_modelpath)
    elif modelpath.is_dir():
        tmp_modelpath = tmp_path
        shutil.copytree(modelpath, tmp_modelpath)
    else:
        raise FileNotFoundError
    return tmp_modelpath


def get_default_dataset_model(
        framework: str) -> Tuple[Type[Dataset], Type[ModelWrapper]]:
    """
    Returns default model and dataset for given framework. Returned dataset is
    a mock of default dataset of returned model.

    Parameters
    ----------
    framework : str
        Name of framework

    Returns
    -------
    Tuple[Type[Dataset], Type[ModelWrapper]] :
        Tuple with dataset and model for given framework
    """
    if framework == 'keras':
        dataset = get_dataset_random_mock(PetDataset)
        modelpath = copy_model_to_tmp(
            TensorFlowPetDatasetMobileNetV2.pretrained_modelpath
        )
        model = TensorFlowPetDatasetMobileNetV2(
            modelpath,
            dataset,
            from_file=True
        )

    elif framework == 'tensorflow':
        dataset = get_dataset_random_mock(MagicWandDataset)
        modelpath = get_tmp_path()
        keras_model = load_model(
            MagicWandModelWrapper.pretrained_modelpath,
            compile=False
        )
        keras_model.save(modelpath)
        model = MagicWandModelWrapper(modelpath, dataset, from_file=True)

    elif framework == 'tflite':
        dataset = get_dataset_random_mock(MagicWandDataset)
        modelpath = copy_model_to_tmp(
            KENNING_MODELS_PATH / 'classification/magic_wand.tflite'
        )
        model = MagicWandModelWrapper(modelpath, dataset, from_file=True)

    elif framework == 'onnx':
        dataset = get_dataset_random_mock(COCODataset2017)
        modelpath = copy_model_to_tmp(ONNXYOLOV4.pretrained_modelpath)
        shutil.copy(
            ONNXYOLOV4.pretrained_modelpath.with_suffix('.cfg'),
            modelpath.with_suffix('.cfg')
        )
        model = ONNXYOLOV4(modelpath, dataset)

    elif framework == 'torch':
        dataset = get_dataset_random_mock(PetDataset)
        modelpath = copy_model_to_tmp(
            PyTorchPetDatasetMobileNetV2.pretrained_modelpath
        )
        model = PyTorchPetDatasetMobileNetV2(
            modelpath,
            dataset=dataset,
            from_file=True
        )
        # save whole model instead of state dict
        model.save_model(modelpath, export_dict=False)

    elif framework == 'darknet':
        dataset = get_dataset_random_mock(COCODataset2017)
        modelpath = copy_model_to_tmp(
            TVMDarknetCOCOYOLOV3.pretrained_modelpath
        )
        model = TVMDarknetCOCOYOLOV3(modelpath, dataset)

    else:
        raise UnknownFramework(f'Unknown framework: {framework}')

    model.save_io_specification(model.modelpath)
    return dataset, model


def remove_file_or_dir(path: str):
    """
    Removes directory of given path

    Parameters
    ----------
    path : str
        Path of given directory or file
    """
    if Path(path).is_file():
        os.remove(path)
    elif Path(path).is_dir():
        shutil.rmtree(path)


def get_dataset_download_path(dataset_cls: Type[Dataset]) -> Path:
    """
    Returns temporary download path for given dataset.

    Parameters
    ----------
    dataset_cls : Type[Dataset]
        Given dataset class

    Returns
    -------
    Path :
        Temporary path for dataset download
    """
    return pytest.test_directory / 'datasets' / dataset_cls.__name__


def get_reduced_dataset_path(dataset_cls: Type[Dataset]) -> Path:
    """
    Returns path to reduced dataset added to docker image.

    Parameters
    ----------
    dataset_cls : Type[Dataset]
        Given dataset class

    Returns
    -------
    Path :
        Path to reduced dataset
    """
    return pytest.test_directory / 'datasets-reduced' / dataset_cls.__name__


def get_dataset_random_mock(dataset_cls: Type[Dataset]) -> Dataset:
    """
    Return a mock for given dataset class.

    Parameters
    ----------
    dataset_cls : Type[Dataset]
        Dataset class to be mocked

    Returns
    -------
    Dataset :
        Mock of given dataset class
    """

    if dataset_cls is PetDataset:
        return RandomizedClassificationDataset(
            get_tmp_path(),
            samplescount=37*4,
            numclasses=37,
            inputdims=(224, 224, 3)
        )
    if dataset_cls is ImageNetDataset:
        return RandomizedClassificationDataset(
            get_tmp_path(),
            samplescount=8,
            numclasses=1000,
            inputdims=(224, 224, 3)
        )
    if dataset_cls is MagicWandDataset:
        dataset = RandomizedClassificationDataset(
            get_tmp_path(),
            samplescount=4*8,
            numclasses=4,
            inputdims=(128, 3, 1)
        )
        return dataset
    if dataset_cls is COCODataset2017:
        return RandomizedDetectionSegmentationDataset(
            get_tmp_path(),
            samplescount=8,
            numclasses=80,
            inputdims=(3, 608, 608)
        )
    raise NotImplementedError


class UnknownFramework(ValueError):
    """
    Raised when unknown framework is passed.
    """
    pass
