# import kenning.resources.models
import sys
if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path
from kenning.utils.class_loader import load_class
from kenning.modelwrappers.classification.pytorch_pet_dataset import PyTorchPetDatasetMobileNetV2  # noqa: E501


class TestModelWrapperAndDatasetCompatibility:
    dataset_dict = {
        'random':
            'random_dataset.RandomizedClassificationDataset',
        'pet_dataset':
            'pet_dataset.PetDataset',
        'open_images':
            'open_images_dataset.OpenImagesDatasetV6',
    }

    def test_one(self, fake_images):
        """
        Test to check if dataset basic functionality works fine.
        """

        def test_whole_functionality(module_name, fake_images):
            import types

            fake_images = fake_images[0]
            dataset = load_class('kenning.datasets.'+module_name)(fake_images)

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

        for module in self.dataset_dict.values():
            test_whole_functionality(module, fake_images)

    def test_two(self, fake_images):
        """
        Test to check basic functionality of ModelWrappers.
        """
        fake_images = fake_images[0]
        modelwrapper_dict = {
            'Pytorch_pet_dataset': [
                'classification.pytorch_pet_dataset.PyTorchPetDatasetMobileNetV2',  # noqa: E501
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
            wrapper

        for modelwrapper_value in modelwrapper_dict.values():
            test_model_wrapper(*modelwrapper_value)

    def test_without_model(self, fake_images):
        """
        Perform tests on ModelWrapper without and model
        """

        images_path, images_amount = fake_images
        module_name = self.dataset_dict['pet_dataset']
        dataset = load_class('kenning.datasets.'+module_name)(images_path)
        modelwrapper = PyTorchPetDatasetMobileNetV2(
            "",
            dataset,
            from_file=False
        )

        measurments = modelwrapper.test_inference()
        assert measurments.data['total'] == images_amount

        print(dataset.get_input_mean_std())
        assert 0
