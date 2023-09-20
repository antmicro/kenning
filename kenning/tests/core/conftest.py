# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from pathlib import Path
from typing import Tuple, Type, Union

import pytest
from tensorflow.keras.models import load_model as load_keras_model
from torch import save as torch_save

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.datasets.coco_dataset import COCODataset2017
from kenning.datasets.imagenet_dataset import ImageNetDataset
from kenning.datasets.magic_wand_dataset import MagicWandDataset
from kenning.datasets.pet_dataset import PetDataset
from kenning.datasets.random_dataset import (
    RandomizedDetectionSegmentationDataset,
)
from kenning.datasets.random_dataset import RandomizedClassificationDataset
from kenning.datasets.visual_wake_words_dataset import VisualWakeWordsDataset
from kenning.modelwrappers.classification.tflite_magic_wand import (
    MagicWandModelWrapper,
)
from kenning.modelwrappers.object_detection.darknet_coco import (
    TVMDarknetCOCOYOLOV3,
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
    ResourceURI :
        URI to the model copy.
    """
    tmp_path = get_tmp_path()
    if model_path.is_file():
        tmp_model_path = tmp_path.with_suffix(model_path.suffix)
        shutil.copy(model_path, tmp_model_path)

        json_path = model_path.with_suffix(model_path.suffix + '.json')
        if json_path.exists():
            shutil.copy(
                json_path,
                tmp_model_path.with_suffix(tmp_model_path.suffix + '.json')
            )

        config_file = model_path.with_suffix('.cfg')
        if config_file.exists():
            shutil.copy(
                config_file,
                tmp_model_path.with_suffix('.cfg')
            )
    elif model_path.is_dir():
        tmp_model_path = tmp_path
        shutil.copytree(model_path, tmp_model_path)
    else:
        raise FileNotFoundError
    return ResourceURI(tmp_model_path)


def get_default_dataset_model(
        framework: str) -> Tuple[Dataset, ModelWrapper]:
    """
    Returns default model and dataset for given framework. Returned dataset is
    a mock of default dataset of returned model.

    Parameters
    ----------
    framework : str
        Name of framework.

    Returns
    -------
    Tuple[Type[Dataset], Type[ModelWrapper]] :
        Tuple with dataset and model for given framework.
    """
    if framework == 'keras':
        dataset = get_dataset_random_mock(MagicWandDataset)
        model_path = copy_model_to_tmp(
            ResourceURI(MagicWandModelWrapper.pretrained_model_uri)
        )
        model = MagicWandModelWrapper(
            model_path,
            dataset,
            from_file=True
        )

    elif framework == 'tensorflow':
        dataset = get_dataset_random_mock(MagicWandDataset)
        model_path = get_tmp_path()
        keras_model = load_keras_model(
            ResourceURI(MagicWandModelWrapper.pretrained_model_uri),
            compile=False
        )
        keras_model.save(model_path)
        model = MagicWandModelWrapper(model_path, dataset, from_file=True)

    elif framework == 'tflite':
        dataset = get_dataset_random_mock(MagicWandDataset)
        model_path = copy_model_to_tmp(
            ResourceURI(MagicWandModelWrapper.pretrained_model_uri)
            .with_suffix('.tflite')
        )
        model = MagicWandModelWrapper(model_path, dataset, from_file=True)

    elif framework == 'onnx':
        dataset = get_dataset_random_mock(MagicWandDataset)
        model_path = get_tmp_path(suffix='.onnx')
        onnx_compiler = ONNXCompiler(dataset, model_path)
        onnx_compiler.compile(
            ResourceURI(MagicWandModelWrapper.pretrained_model_uri)
        )
        model = MagicWandModelWrapper(model_path, dataset, from_file=True)

    elif framework == 'torch':
        import dill
        dataset = get_dataset_random_mock(MagicWandDataset)
        onnx_model_path = get_tmp_path(suffix='.onnx')
        onnx_compiler = ONNXCompiler(dataset, onnx_model_path)
        onnx_compiler.compile(
            ResourceURI(MagicWandModelWrapper.pretrained_model_uri)
        )

        torch_model = onnx2torch.convert(onnx_model_path)

        model_path = get_tmp_path(suffix='.pth')
        torch_save(torch_model, model_path, pickle_module=dill)

        model = MagicWandModelWrapper(model_path, dataset, from_file=True)

    elif framework == 'darknet':
        dataset = get_dataset_random_mock(COCODataset2017)
        model_path = copy_model_to_tmp(
            ResourceURI(TVMDarknetCOCOYOLOV3.pretrained_model_uri)
        )
        model = TVMDarknetCOCOYOLOV3(model_path, dataset)

    elif framework == 'iree':
        dataset = get_dataset_random_mock(MagicWandDataset)
        model_path = get_tmp_path(suffix='.vmfb')
        iree_compiler = IREECompiler(dataset, model_path)
        iree_compiler.compile(
            ResourceURI(MagicWandModelWrapper.pretrained_model_uri)
        )
        model = MagicWandModelWrapper(model_path, dataset, from_file=True)

    elif framework == 'tvm':
        dataset = get_dataset_random_mock(MagicWandDataset)
        model_path = get_tmp_path(suffix='.tar')
        tvm_compiler = TVMCompiler(
            dataset, model_path, model_framework='keras'
        )
        tvm_compiler.compile(
            ResourceURI(MagicWandModelWrapper.pretrained_model_uri)
        )
        model = MagicWandModelWrapper(model_path, dataset, from_file=True)

    else:
        raise UnknownFramework(f'Unknown framework: {framework}')

    model.save_io_specification(model.model_path)
    return dataset, model


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


def get_dataset_download_path(dataset_cls: Type[Dataset]) -> Path:
    """
    Returns temporary download path for given dataset.

    Parameters
    ----------
    dataset_cls : Type[Dataset]
        Given dataset class.

    Returns
    -------
    Path :
        Temporary path for dataset download.
    """
    return pytest.test_directory / 'datasets' / dataset_cls.__name__


def get_reduced_dataset_path(dataset_cls: Type[Dataset]) -> Path:
    """
    Returns path to reduced dataset added to docker image.

    Parameters
    ----------
    dataset_cls : Type[Dataset]
        Given dataset class.

    Returns
    -------
    Path :
        Path to reduced dataset.
    """
    return pytest.test_directory / dataset_cls.__name__


def get_dataset_random_mock(dataset_cls: Type[Dataset]) -> Dataset:
    """
    Return a mock for given dataset class.

    Parameters
    ----------
    dataset_cls : Type[Dataset]
        Dataset class to be mocked.

    Returns
    -------
    Dataset :
        Mock of given dataset class.
    """

    if dataset_cls is PetDataset:
        return RandomizedClassificationDataset(
            get_tmp_path(),
            samplescount=37*5,
            numclasses=37,
            inputdims=(224, 224, 3)
        )
    if dataset_cls is ImageNetDataset:
        return RandomizedClassificationDataset(
            get_tmp_path(),
            samplescount=8*5,
            numclasses=8,
            inputdims=(224, 224, 3)
        )
    if dataset_cls is MagicWandDataset:
        dataset = RandomizedClassificationDataset(
            get_tmp_path(),
            samplescount=4*10,
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
    if dataset_cls is VisualWakeWordsDataset:
        return RandomizedClassificationDataset(
            get_tmp_path(),
            samplescount=10,
            numclasses=2,
            inputdims=(480, 320, 3)
        )
    raise NotImplementedError


class UnknownFramework(ValueError):
    """
    Raised when unknown framework is passed.
    """
    pass
