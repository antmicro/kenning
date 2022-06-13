import pytest
import kenning
from random import randint, random, seed
from pathlib import Path
from PIL import Image
from kenning.utils.class_loader import load_class
from kenning.core.dataset import Dataset
seed(12345)


def write_to_dirs(dir_path, amount):
    """
    Creates files under provided 'dir_path' such as 'list.txt' for PetDataset,
    'annotations.csv' and 'classnames.csv' for OpenImagesDataset.
    """
    def three_random_one_hot(i):
        return f'{i%37+1} {randint(0, 1)} {randint(0, 1)}'

    def four_random():
        return f'{random()},{random()},{random()},{random()}'

    with open(dir_path / 'annotations' / 'list.txt', 'w') as f:
        [print(f'image_{i} {three_random_one_hot(i)}', file=f)
         for i in range(amount)]
    with open(dir_path / 'classnames.csv', 'w') as f:
        print('/m/o0fd,person', file=f)
    with open(dir_path / 'annotations.csv', 'w') as f:
        title = 'ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,'
        title += 'IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside'
        print(title, file=f)
        [print(f'image_{i},xclick,/m/o0fd,1,{four_random()},0,0,0,0,0', file=f)
         for i in range(amount)]
    return


@pytest.fixture(scope='session')
def modelwrapperSamples(fake_images, datasetSamples):
    class WrapperData:
        kenning_path = kenning.__path__[0]

        def __init__(self):
            torch_pet_mobilenet_import_path = "kenning.modelwrapper.classification.pytorch_pet_dataset.PytorchPetDatasetMobileNetV2"    # noqa: E501
            torch_pet_mobilenet_model_path = self.kenning_path + "/resources/models/classification/pytorch_pet_dataset_mobilenetv2.pth"   # noqa: E501
            tensorflow_pet_mobilenet_import_path = "kenning.modelwrapper.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2"  # noqa: E501
            tensorflow_pet_mobilenet_model_path = self.kenning_path + "/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5"    # noqa: E501

            self.pytorch_pet_mobilenetv2 = self.init_modelwrapper(
                torch_pet_mobilenet_import_path,
                dataset=datasetSamples.pet_dataset,
                model=torch_pet_mobilenet_model_path)

            self.tensorflow_pet_mobilenetv2 = self.init_modelwrapper(
                tensorflow_pet_mobilenet_import_path,
                dataset=datasetSamples.pet_dataset,
                model=tensorflow_pet_mobilenet_model_path
            )

        def init_modelwrapper(self,
                              import_path: str,
                              dataset: Dataset = None,
                              model: str = "",
                              from_file: bool = False):
            wrapper = load_class(import_path)
            wrapper = wrapper(model, dataset, from_file=from_file)
            return wrapper

    return WrapperData()


@pytest.fixture()
def datasetSamples(fake_images):
    class DatasetData:
        def __init__(self):
            self.pet_dataset = self.init_pet_dataset()

        def init_pet_dataset(self, datapath: Path = fake_images.path):
            pet_dataset = load_class("kenning.datasets.pet_dataset.PetDataset")
            pet_dataset = pet_dataset(datapath)
            return pet_dataset

    return DatasetData()


@pytest.fixture
def empty_dir(tmp_path):
    (tmp_path / 'annotations').mkdir()
    (tmp_path / 'annotations' / 'list.txt').touch()
    return tmp_path


@pytest.fixture
def fake_images(empty_dir):
    """
    Creates a temporary dir with images.

    Images are located under 'image/' folder.
    """
    amount = 148
    write_to_dirs(empty_dir, amount)
    (empty_dir / 'images').mkdir()
    (empty_dir / 'img').symlink_to(empty_dir / 'images')
    for i in range(amount):
        file = (empty_dir / 'images' / f'image_{i}.jpg')
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        img = Image.new(mode='RGB', size=(5, 5), color=color)
        img.save(file, 'JPEG')

    class DataFolder:
        def __init__(self, datapath: Path, amount: int):
            self.path = datapath
            self.amount = amount

    return DataFolder(empty_dir, amount)
