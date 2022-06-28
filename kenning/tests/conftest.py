import pytest
import shutil
import kenning
import tempfile
from random import randint, random
from pathlib import Path
from PIL import Image
from kenning.utils.class_loader import load_class
from kenning.core.dataset import Dataset
from dataclasses import dataclass


@dataclass
class DataFolder:
    """
    A dataclass to store datasetimages fixture properties.

    Parameters
    --------
    path: Path
        A path to data files
    amount: int
        Amount of generated images
    """
    path: Path
    amount: int


class Samples:
    def __init__(self):
        """
        The base class for object samples.
        """
        self._data_index = 0
        self.samples = {}
        self.kenning_path = kenning.__path__[0]
        pass

    def get(self, data_name: str):
        """
        Returns data for specified key.

        Parameters
        ----------
        data_name: str
            A key for the sample.

        Returns
        -------
        Any:
            Data associated with provided sample.
        """
        return self.samples[data_name]

    def __iter__(self):
        """
        Provides iterator over data samples.

        Returns
        -------
        Samples:
            this object.
        """
        self._data_index = 0
        self._samples = tuple(self.samples.values())
        return self

    def __next__(self):
        """
        Returns next object sample.

        Returns
        -------
        Any:
            object sample.
        """
        if self._data_index < len(self._samples):
            prev = self._data_index
            self._data_index += 1
            return self._samples[prev]
        raise StopIteration


@pytest.fixture()
def modelsamples():
    class ModelData(Samples):
        def __init__(self):
            """
            Model samples.
            Stores paths to models presented in Kenning docs.
            """
            super().__init__()
            self.init_model("/resources/models/classification/pytorch_pet_dataset_mobilenetv2_full_model.pth",  # noqa: E501
                            'torch',
                            'PyTorchPetDatasetMobileNetV2')
            self.init_model("/resources/models/classification/pytorch_pet_dataset_mobilenetv2.pth",  # noqa: E501
                            'torch_weights',
                            'PyTorchPetDatasetMobileNetV2')
            self.init_model("/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5",           # noqa: E501
                            'keras',
                            'TensorFlowPetDatasetMobileNetV2')

        def init_model(self, model_path: str,
                       modelframework: str,
                       modelwrapper: str):
            """
            Adds path to model with associated framework
            and associated modelwrapper name to samples

            Parameters
            ----------
            model_path: str
                The path to the model (Relative to kenning's directory).
            modelframework: str
                The framework model is compatible with.
            modelwrapper: str
                The name of ModelWrapper that is compatible with model.

            Returns
            -------
            Tuple[str, str]:
                Returns tuple with absolute path to model
                and framework it compatible with.
            """
            model_path = self.kenning_path + model_path
            self.samples[modelframework] = (model_path, modelwrapper)
            return (model_path, modelwrapper)
    return ModelData()


@pytest.fixture()
def optimizersamples(datasetimages: DataFolder, datasetsamples: Samples):
    class OptimizerData(Samples):
        def __init__(self):
            """
            Optimizer samples.
            Stores basic Optimizer objects with its parameters.
            """
            super().__init__()
            self.init_optimizer('kenning.compilers.tflite.TFLiteCompiler',
                                'default',
                                'keras',
                                'tflite',
                                dataset=datasetsamples.get('PetDataset'),
                                compiled_model_path=datasetimages.path)

            self.init_optimizer('kenning.compilers.tvm.TVMCompiler',
                                'llvm',
                                'keras',
                                'so',
                                dataset=datasetsamples.get('PetDataset'),
                                compiled_model_path=datasetimages.path)

            self.init_optimizer('kenning.compilers.tvm.TVMCompiler',
                                'llvm',
                                'torch',
                                'so',
                                dataset=datasetsamples.get('PetDataset'),
                                compiled_model_path=datasetimages.path)

        def init_optimizer(self,
                           import_path: str,
                           target: str,
                           modelframework: str,
                           filesuffix: str,
                           dataset: Dataset = None,
                           compiled_model_path: Path = datasetimages.path,
                           dataset_percentage: float = 1.0,
                           **kwargs):
            """
            Initializes Optimizer with its compilation arguments
            Adds Optimizer, its target and modelframework to samples.

            Parameters
            ----------
            import_path: str
                The import path optimizer will be imported with.
            target: str
                Target accelerator on which the model will be executed.
            modelframework: str
                Framework of the input model, used to select a proper backend.
            filesuffix: str
                The suffix compiled model should be saved with.
            dataset:Dataset
                Dataset used to train the model - may be used for quantization
                during compilation stage.
            compiled_model_path: Path
                Path where compiled model will be saved.
            dataset_percentage: float
                If the dataset is used for optimization (quaantization), the
                dataset percentage determines how much of data samples is going
                to be used.

            Returns
            -------
            Optimizer:
                Initialized optimizer object.
            """
            optimizer_name = import_path.rsplit('.')[-1] + '_' + modelframework
            file_name = optimizer_name + '.' + filesuffix
            compiled_model_path = compiled_model_path / file_name
            optimizer = load_class(import_path)
            optimizer = optimizer(dataset,
                                  compiled_model_path,
                                  target=target,
                                  modelframework=modelframework,
                                  dataset_percentage=dataset_percentage,
                                  **kwargs)
            self.samples[optimizer_name] = optimizer
            return optimizer
    return OptimizerData()


@pytest.fixture()
def modelwrappersamples(datasetsamples: Samples, modelsamples: Samples):
    class WrapperData(Samples):
        def __init__(self):
            """
            ModelWrapper samples.
            Stores basic ModelWrapper objects.
            """
            super().__init__()
            torch_pet_mobilenet_import_path = "kenning.modelwrappers.classification.pytorch_pet_dataset.PyTorchPetDatasetMobileNetV2"    # noqa: E501
            tensorflow_pet_mobilenet_import_path = "kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2"  # noqa: E501

            self.init_modelwrapper(
                torch_pet_mobilenet_import_path,
                modelsamples.get('torch_weights')[0],
                dataset=datasetsamples.get('PetDataset')
            )

            self.init_modelwrapper(
                tensorflow_pet_mobilenet_import_path,
                modelsamples.get('keras')[0],
                dataset=datasetsamples.get('PetDataset')
            )

        def init_modelwrapper(self,
                              import_path: str,
                              model: str,
                              dataset: Dataset = None,
                              from_file: bool = True,
                              **kwargs):
            """
            Initializes ModelWrapper and adds it to samples.

            Parameters
            ----------
            import_path: str
                The import path modelwrapper will be imported with.
            model: str
                The path to modelwrapper's model.
            dataset: Dataset
                The dataset to verify inference.
            from_file: bool
                True if model should be loaded from file.

            Returns
            -------
            ModelWrapper:
                initialized ModelWrapper.
            """
            wrapper = load_class(import_path)
            wrapper = wrapper(model, dataset, from_file=from_file, **kwargs)
            modelwrapper_name = import_path.rsplit('.')[-1]
            self.samples[modelwrapper_name] = wrapper
            return wrapper
    return WrapperData()


@pytest.fixture()
def datasetsamples(datasetimages: DataFolder):
    class DatasetData(Samples):
        def __init__(self):
            """
            Dataset samples.
            Stores basic dataset objects.
            """
            super().__init__()
            self.init_dataset("kenning.datasets.pet_dataset.PetDataset")

        def init_dataset(self,
                         import_path: str,
                         datapath: Path = datasetimages.path,
                         batch_size: int = 1,
                         download_dataset: bool = False,
                         **kwargs):
            """
            Initializes Dataset and adds it to samples.

            Parameters
            ----------
            import_path: str
                The import path dataset will be imported with.
            datapath: Path
                The path to dataset data.
            batch_size: int
                The dataset batch size.
            download_dataset: bool
                True if dataset should be downloaded first.

            Returns
            -------
            Dataset: initialized Dataset.
            """
            dataset = load_class(import_path)
            dataset = dataset(datapath,
                              batch_size=batch_size,
                              download_dataset=download_dataset,
                              **kwargs)
            dataset_name = import_path.rsplit('.')[-1]
            self.samples[dataset_name] = dataset
            return dataset
    return DatasetData()


@pytest.fixture(scope='class')
def datasetimages():
    """
    Creates a temporary dir with images and data files.
    Images are located under 'image/' folder.

    Returns
    -------
    DataFolder: A DataFolder object that stores path to data and images amount
    """
    images_amount = 148
    path = Path(tempfile.NamedTemporaryFile().name)
    path.mkdir()
    (path / 'images').mkdir()
    (path / 'img').symlink_to(path / 'images')
    (path / 'annotations').mkdir()
    (path / 'annotations' / 'list.txt').touch()
    write_to_dirs(path, images_amount)

    for i in range(images_amount):
        file = (path / 'images' / f'image_{i}.jpg')
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        img = Image.new(mode='RGB', size=(5, 5), color=color)
        img.save(file, 'JPEG')

    yield DataFolder(path, images_amount)
    shutil.rmtree(path)


def write_to_dirs(path, amount):
    """
    Creates files under provided 'path' such as 'list.txt' for PetDataset,
    'annotations.csv' and 'classnames.csv' for OpenImagesDataset.

    Parameters
    --------
    path: Path
        The Path to where data have to be located
    amount: int
        Amount of images are being written to data files
    """
    def three_random_one_hot(i):
        return f'{i%37+1} {randint(0, 1)} {randint(0, 1)}'

    def four_random():
        return f'{random()},{random()},{random()},{random()}'

    with open(path / 'annotations' / 'list.txt', 'w') as f:
        [print(f'image_{i} {three_random_one_hot(i)}', file=f)
         for i in range(amount)]

    with open(path / 'classnames.csv', 'w') as f:
        print('/m/o0fd,person', file=f)

    with open(path / 'annotations.csv', 'w') as f:
        title = 'ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,'
        title += 'IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside'
        print(title, file=f)
        [print(f'image_{i},xclick,/m/o0fd,1,{four_random()},0,0,0,0,0', file=f)
            for i in range(amount)]
