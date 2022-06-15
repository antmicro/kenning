class TestModelWrapperAndDatasetCompatibility:
    def test_deliver_input(self, datasetSamples):
        """
        Tests dataset functions used by modelwrappers for data delivering

        List of methods are being tested
        --------------------------------
        dataset.prepare_input_samples()
        dataset.prepare_output_samples()
        dataset.get_input_mean_std()
        dataset.train_test_split_representations()

        Used fixtures
        -------------
        datasetSamples - to get dataset instances.
        """

        def run_tests(dataset):
            # Test train_test_split_representations
            # this test should check the return type
            # if train or validation data is empty
            # and check that corresponding sets of data are equal in length
            print(dataset)
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
            mean_and_std = dataset.get_input_mean_std()
            assert isinstance(mean_and_std, tuple)

        for dataset in datasetSamples:
            run_tests(dataset)

    def test_deliver_output(self, fake_images, modelwrapperSamples):
        """
        Tests modelwrapper functions to deliver output to datasets

        List of methods are being tested
        --------------------------------
        modelwrapper.test_inference()
        dataset.evaluate()

        Used fixtures
        -------------
        fake_images - to get total amount of images.
        modelwrapperSamples - to get modelwrapper instances.
        """

        def run_tests(wrapper):
            from kenning.core.measurements import Measurements

            # Test modelwrapper.test_inference (includes dataset.evaluate)
            # this test has to check if output is an istance of Measurements()
            # and does the amount of counted images equals to provided ones
            measurements = wrapper.test_inference()
            assert isinstance(measurements, Measurements)
            assert measurements.get_values('total') == fake_images.amount

        for wrapper in modelwrapperSamples:
            run_tests(wrapper)
