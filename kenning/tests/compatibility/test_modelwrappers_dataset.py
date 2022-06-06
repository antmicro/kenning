from kenning.utils.class_loader import load_class


class TestModelWrapperAndDatasetCompatibility:
    dataset_dict = {
        'pet_dataset':
            'pet_dataset.PetDataset',
    }

    modelwrapper_dict = {
        'classification_pytorch_pet_dataset':
        [
            'classification.pytorch_pet_dataset.PyTorchPetDatasetMobileNetV2',
            'pet_dataset.PetDataset',
        ],
        'classification_tensorflow_pet_dataset':
        [
            'classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2',  # noqa: E501
            'pet_dataset.PetDataset'
        ],
    }

    def test_deliver_input(self, fake_images):
        """
        Tests dataset functions used by modelwrappers for data delivering

        List of methods are being tested
        -----------------------
        dataset.prepare_input_samples
        dataset.prepare_output_samples
        dataset.get_input_mean_std
        dataset.train_test_split_representations

        Used fixtures
        -------------
        fake_images - to generate images and feed them to datasets
        """

        def run_tests(dataset, images_path):
            dataset = load_class('kenning.datasets.' + dataset)(images_path)

            # Test train_test_split_representations
            # this test should check the return type
            # if train or validation data is empty
            # and check that corresponding sets of data are equal in length
            Xt, Xv, Yt, Yv = dataset.train_test_split_representations(
                test_fraction=0.25)
            assert isinstance(Xt, list) and isinstance(Xv, list)
            assert len(Xt) > 0 and len(Xv) > 0
            assert len(Xt) == len(Yt) and len(Xv) == len(Yv)

            # Test prepare_input_samples
            # this test should check the return type
            # if length of output equals to length of provided data
            samples = dataset.prepare_input_samples(Xv)
            assert isinstance(samples, list)
            assert len(samples) == len(Xv)

            # Test prepare_output_samples
            # this test should check the return type
            # if length of output equals to length of provided data

            samples = dataset.prepare_output_samples(Yv)
            assert isinstance(samples, list)
            assert len(samples) == len(Yv)

            # TODO: This method isn't implemented for openimages dataset
            # what leads to NotImplementedError, to my opnion
            # method should be cutted off from base class
            # or implementation should be added to openimages dataset

            # Test get_input_mean_std
            # this test has just check if output of method is a numerical value
            mean, std = dataset.get_input_mean_std()

        for dataset in self.dataset_dict.values():
            run_tests(dataset, fake_images[0])

    def test_deliver_output(self, fake_images):
        """
        Tests modelwrapper functions to deliver output to datasets

        List of methods are being tested
        -----------------------
        modelwrapper.test_inference
        dataset.evaluate

        Used fixtures
        -------------
        fake_images - to generate images and feed them to datasets
        """

        def run_tests(wrapper_path, dataset_path, images_path, images_count):
            datasetcls = load_class("kenning.datasets."+dataset_path)
            dataset = datasetcls(images_path)
            modelwrappercls = load_class("kenning.modelwrappers."+wrapper_path)
            wrapper = modelwrappercls(images_path, dataset, from_file=False)

            measurements = wrapper.test_inference()
            print(measurements)

        for wrapper, dataset in self.modelwrapper_dict.values():
            run_tests(wrapper, dataset, fake_images[0], fake_images[1])
