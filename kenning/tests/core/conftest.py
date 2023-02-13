from typing import Final, Tuple, Type
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


KENNING_MODELS_PATH: Final = Path(r'kenning/resources/models/')
# use only 10% of original dataset to save time
RANDOM_DATASET_SAMPLES: Final = 256


def get_tmp_path() -> Path:
    """
    Generates temporary path

    Returns
    -------
    Path :
        Temporary path
    """
    return Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())


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
    if os.path.isfile(modelpath):
        tmp_modelpath = get_tmp_path().with_suffix(modelpath.suffix)
        shutil.copy(modelpath, tmp_modelpath)
    elif os.path.isdir(modelpath):
        tmp_modelpath = get_tmp_path()
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
        modelpath = get_tmp_path().with_suffix('.pb')
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
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def get_dataset_download_path(dataset_cls: Type[Dataset]) -> Path:
    """
    Returns temporary download path for given dataset

    Parameters
    ----------
    dataset_cls : Type[Dataset]
        Given dataset class

    Returns
    -------
    Path :
        Temporary path for dataset download
    """
    return Path(f'/tmp/{dataset_cls.__name__}')


def get_dataset(dataset_cls: Type[Dataset]) -> Dataset:
    """
    Returns dataset instance of given class. It tries to used already download
    data and if that fails it download the data.

    Parameters
    ----------
    dataset_cls : Type[Dataset]
        Class of the dataset to be returned

    Returns
    -------
    Dataset :
        Instance of given dataset class
    """
    download_path = get_dataset_download_path(dataset_cls)
    try:
        dataset = dataset_cls(download_path, download_dataset=False)
    except FileNotFoundError:
        dataset = dataset_cls(download_path, download_dataset=True)
    except Exception:
        raise
    return dataset


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
        class RndClassDatasetCopy(RandomizedClassificationDataset):
            pass
        RndClassDatasetCopy.train_test_split_representations = \
            MagicWandDataset.train_test_split_representations
        RndClassDatasetCopy.prepare_tf_dataset = \
            MagicWandDataset.prepare_tf_dataset
        dataset = RndClassDatasetCopy(
            get_tmp_path(),
            samplescount=4*8,
            numclasses=4,
            integer_classes=True,
            inputdims=(128, 3, 1)
        )
        dataset.dataX = dataset.prepare_input_samples(dataset.dataX)
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
