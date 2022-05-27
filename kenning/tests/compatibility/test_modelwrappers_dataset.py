import kenning.modelwrappers.classification.pytorch_pet_dataset as wrappers
import kenning.resources.models
import sys
if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path
from importlib import import_module


class TestModelWrapperAndDatasetCompatibility:
    def test_deliver_input(self, fake_images):
        """
        Test to check if dataset's output is delivered
        to ModelWrapper. That means that we have to test datasets here.

        Test .train_test_split_representations() method.
        """
        dataset_dict = {
            'random': [
                'random_dataset',
                'RandomizedClassificationDataset',
                fake_images,
            ],
            'pet_dataset': [
                'pet_dataset',
                'PetDataset',
                fake_images,
            ],
            'open_images': [
                'open_images_dataset',
                'OpenImagesDatasetV6',
                fake_images,
            ],
        }

        def test_random(module_name, module_package, *args):
            import types
            import typing

            module = import_module('kenning.datasets.' + module_name)
            dataset = getattr(module, module_package)(*args, batch_size=1)

            Xt, Xv, Yt, Yv = dataset.train_test_split_representations(0.25)

            def get_length(item):
                length = 0
                if (len(item) == 0):
                    return length
                elif (isinstance(item[0], list)):
                    for List in item:
                        length += get_length(List)
                return length

            # Check if length is equal for corresponding data
            assert get_length(Xt) == get_length(Yt)
            assert get_length(Xv) == get_length(Yv)

            # Check generator datatype and does it throw an error
            generator = dataset.calibration_dataset_generator()
            assert isinstance(generator, types.GeneratorType)
            assert next(generator)

            # Check if dataset's numclasses equals to class_names length
            class_names = dataset.get_class_names()
            assert len(class_names) == dataset.numclasses
            assert issubclass(class_names, list) and class_names[0] is str
            return dataset

        for dataset_value in dataset_dict.values():
            test_random(*dataset_value)

    # def test_without_dataset(self):
    #     """
    #     Perform tests on ModelWrapper with provided dataset as None
    #     """
    #     modelpath = str(path(kenning.resources.models, "classification")) +\
    #         "/pytorch_pet_dataset_mobilenetv2.pth"
    #     assert 0
    #     wrappers.PyTorchPetDatasetMobileNetV2(    # noqa: E501
    #         modelpath,
    #         None
    #     )
