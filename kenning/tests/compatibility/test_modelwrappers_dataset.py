from kenning.utils.class_loader import load_class


class TestModelWrapperAndDatasetCompatibility:
    dataset_dict = {
        'pet_dataset':
            'pet_dataset.PetDataset',
        'open_images':
            'open_images_dataset.OpenImagesDatasetV6',
    }
    # 'random':
    #        'random_dataset.RandomizedClassificationDataset',

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

        def run_tests(module, images_path):
            dataset = load_class('kenning.datasets.' + module)(images_path)

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
            # mean, std = dataset.get_input_mean_std()

        for module in self.dataset_dict.values():
            run_tests(module, fake_images[0])
