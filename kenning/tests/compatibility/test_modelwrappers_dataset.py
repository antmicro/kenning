import kenning.resources.models
import sys
if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path
from importlib import import_module
from kenning.utils.class_loader import load_class


class TestModelWrapperAndDatasetCompatibility:
    def test_one(self, fake_images):
        """
        Test to check if dataset basic functionality works fine.
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

        def test_whole_functionality(module_name, module_package, *args):
            import types

            module = import_module('kenning.datasets.' + module_name)
            dataset = getattr(module, module_package)(*args)

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
            # and have right return types
            class_names = dataset.get_class_names()
            assert len(class_names) == dataset.numclasses
            assert isinstance(class_names, list) and isinstance(class_names[0], str)    # noqa: E501

            # OpenImagesDatasetV6 throws NotImplementedError
            # assert isinstance(dataset.get_input_mean_std(), tuple)

            dataset.prepare()

            prepared = dataset.prepare_input_samples(dataset.dataX)
            assert len(prepared) == len(dataset.dataX)
            assert isinstance(prepared, list)

            # Throws an error, most likely have to be used with ModelWrapper
            # prepared = dataset.prepare_output_samples(dataset.dataX)
            # assert isinstance(prepared, list)

            dataset.set_batch_size(10)
            assert dataset.batch_size == 10

            return

        for dataset_value in dataset_dict.values():
            test_whole_functionality(*dataset_value)

    def test_two(self, fake_images):
        """
        Test to check basic functionality of ModelWrappers.
        """
        modelwrapper_dict = {
            'Pytorch_pet_dataset': [
                'classification.pytorch_pet_dataset.PyTorchPetDatasetMobileNetV2',
                ['classification',
                 'pytorch_pet_dataset_mobilenetv2.pth'
                 ],
                'kenning.datasets.pet_dataset.PetDataset',
                fake_images
            ]
        }

        def test_model_wrapper(wrapper_name, model_path, dataset_name, root):
            model_module = 'kenning.resources.models.' + model_path[0]
            model = path(model_module, model_path[1])
            wrapper_name = 'kenning.modelwrappers.' + wrapper_name
            dataset = load_class(dataset_name)(root)
            wrapper = load_class(wrapper_name)(model, dataset, from_file=True)
            # module = import_module(module_name)
            # module_dataset = import_module(dataset_name.rsplit('.', 1)[0])
            # dataset = getattr(module_dataset, root)
            # modelwrapper = getattr(module, module_package)(model, dataset)

        for modelwrapper_value in modelwrapper_dict.values():
            test_model_wrapper(*modelwrapper_value)

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
