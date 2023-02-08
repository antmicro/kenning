from typing import Final, List, Tuple, Type
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
from kenning.datasets.random_dataset import RandomizedDetectionDataset
from kenning.modelwrappers.classification.pytorch_pet_dataset import PyTorchPetDatasetMobileNetV2   # noqa: 501
from kenning.modelwrappers.classification.tflite_magic_wand import MagicWandModelWrapper    # noqa: 501
from kenning.modelwrappers.detectors.yolov4 import ONNXYOLOV4
from kenning.modelwrappers.detectors.darknet_coco import TVMDarknetCOCOYOLOV3


KENNING_MODELS_PATH: Final = Path(r'kenning/resources/models/')
# use only 10% of original dataset to save time
RANDOM_DATASET_SAMPLES: Final = 256


def get_tmp_name() -> str:
    return r'/tmp/' + next(tempfile._get_candidate_names())


def get_all_subclasses(cls: type) -> List[type]:
    result = []
    queue = [cls]
    while queue:
        q = queue.pop()
        if len(q.__subclasses__()) == 0:
            result.append(q)
        for sub_q in q.__subclasses__():
            queue.append(sub_q)
    return result


def get_default_dataset_model(framework: str) -> Tuple[Dataset, ModelWrapper]:
    if framework == 'keras':
        dataset = get_dataset_random_mock(MagicWandDataset)
        modelpath = KENNING_MODELS_PATH / 'classification/magic_wand.h5'
        model = MagicWandModelWrapper(modelpath, dataset, from_file=True)

    elif framework == 'tensorflow':
        dataset = get_dataset_random_mock(MagicWandDataset)
        modelpath = KENNING_MODELS_PATH / 'classification/magic_wand.pb'
        keras_model = load_model(KENNING_MODELS_PATH / 'classification/magic_wand.h5')  # noqa: 501
        keras_model.save(KENNING_MODELS_PATH / 'classification/magic_wand.pb')
        shutil.copy(
            KENNING_MODELS_PATH / 'classification/magic_wand.h5.json',
            KENNING_MODELS_PATH / 'classification/magic_wand.pb.json'
        )
        model = MagicWandModelWrapper(modelpath, dataset, from_file=True)

    elif framework == 'tflite':
        dataset = get_dataset_random_mock(MagicWandDataset)
        modelpath = KENNING_MODELS_PATH / 'classification/magic_wand.tflite'
        model = MagicWandModelWrapper(modelpath, dataset, from_file=True)

    elif framework == 'onnx':
        dataset = get_dataset_random_mock(COCODataset2017)
        modelpath = KENNING_MODELS_PATH / 'detection/yolov4.onnx'
        model = ONNXYOLOV4(modelpath, dataset)

    elif framework == 'torch':
        dataset = get_dataset_random_mock(PetDataset)
        modelpath = KENNING_MODELS_PATH / 'classification/pytorch_pet_dataset_mobilenetv2_full_model.pth'  # noqa: 501
        model = PyTorchPetDatasetMobileNetV2(
            modelpath,
            dataset=dataset,
            from_file=True
        )
        model.save_io_specification(modelpath)

    elif framework == 'darknet':
        dataset = get_dataset_random_mock(COCODataset2017)
        modelpath = KENNING_MODELS_PATH / 'detection/yolov4.cfg'
        shutil.copy(
            KENNING_MODELS_PATH / 'detection/yolov4.onnx.json',
            KENNING_MODELS_PATH / 'detection/yolov4.cfg.json'
        )
        model = TVMDarknetCOCOYOLOV3(modelpath, dataset)

    else:
        raise UnknownFramework(f'Unknown framework: {framework}')

    return dataset, model


def remove_file_or_dir(path: str):
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def get_model_path(cls: Type[ModelWrapper]) -> Path:
    return Path(f'/tmp/{cls.__name__}_{next(tempfile._get_candidate_names())}')


def get_dataset_download_path(cls: Type[Dataset]) -> Path:
    return Path(f'/tmp/{cls.__name__}')


def get_dataset(cls: Type[Dataset]) -> Dataset:
    download_path = get_dataset_download_path(cls)
    try:
        dataset = cls(download_path, download_dataset=False)
    except FileNotFoundError:
        dataset = cls(download_path, download_dataset=True)
    except Exception:
        raise
    return dataset


def get_dataset_random_mock(dataset_cls: Type[Dataset]) -> Type[Dataset]:

    if dataset_cls is PetDataset:
        return RandomizedClassificationDataset(
            get_dataset_download_path(RandomizedClassificationDataset),
            samplescount=37*4,
            numclasses=37,
            inputdims=(224, 224, 3)
        )
    if dataset_cls is ImageNetDataset:
        return RandomizedClassificationDataset(
            get_dataset_download_path(RandomizedClassificationDataset),
            samplescount=8,
            numclasses=1000,
            inputdims=(224, 224, 3)
        )
    if dataset_cls is MagicWandDataset:
        RandomizedClassificationDataset.train_test_split_representations = \
            MagicWandDataset.train_test_split_representations
        RandomizedClassificationDataset.prepare_tf_dataset = \
            MagicWandDataset.prepare_tf_dataset
        dataset = RandomizedClassificationDataset(
            get_dataset_download_path(RandomizedClassificationDataset),
            samplescount=4*8,
            numclasses=4,
            integer_classes=True,
            inputdims=(128, 3, 1)
        )
        dataset.dataX = dataset.prepare_input_samples(dataset.dataX)
        return dataset
    if dataset_cls is COCODataset2017:
        return RandomizedDetectionDataset(
            get_dataset_download_path(RandomizedDetectionDataset),
            samplescount=8,
            numclasses=80,
            inputdims=(3, 608, 608)
        )
    raise NotImplementedError


class UnknownFramework(Exception):
    pass
