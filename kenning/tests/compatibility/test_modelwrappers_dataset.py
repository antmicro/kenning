from kenning.tests.conftest import DataFolder, Samples


class TestModelWrapperAndDatasetCompatibility:
    def test_deliver_input(self, datasetsamples: Samples):
        """
        Tests dataset functions used by modelwrappers for data delivering

        List of methods that are being tested
        --------------------------------
        dataset.prepare_input_samples()
        dataset.prepare_output_samples()
        dataset.get_input_mean_std()
        dataset.train_test_split_representations()

        Used fixtures
        -------------
        datasetsamples - to get dataset instances.
        """

        def run_tests(dataset):
            """
            Parameters
            ---------
            dataset : Dataset
                A Dataset object that is being tested
            """
            Xt, Xv, Yt, Yv = dataset.train_test_split_representations(
                test_fraction=0.25)
            assert isinstance(Xt, list) and isinstance(Xv, list)
            assert len(Xt) > 0 and len(Xv) > 0
            assert len(Xt) == len(Yt) and len(Xv) == len(Yv)

            samples = dataset.prepare_input_samples(Xv)
            assert isinstance(samples, list)
            assert len(samples) == len(Xv)

            samples = dataset.prepare_output_samples(Yv)
            assert isinstance(samples, list)
            assert len(samples) == len(Yv)

            mean_and_std = dataset.get_input_mean_std()
            assert isinstance(mean_and_std, tuple)

        for dataset in datasetsamples:
            run_tests(dataset)

    def test_deliver_output(self,
                            datasetimages: DataFolder,
                            modelwrappersamples: Samples):
        """
        Tests modelwrapper functions to deliver output to datasets

        List of methods that are being tested
        --------------------------------
        modelwrapper.test_inference()
        dataset.evaluate()

        Used fixtures
        -------------
        datasetimages - to get total amount of images.
        modelwrappersamples - to get modelwrapper instances.
        """

        def run_tests(wrapper):
            """
            Parameters
            ---------
            wrapper: ModelWrapper
                A ModelWrapper object that is being tested
            """
            from kenning.core.measurements import Measurements

            measurements = wrapper.test_inference()
            assert isinstance(measurements, Measurements)
            assert measurements.get_values('total') == datasetimages.amount

        for wrapper in modelwrappersamples:
            run_tests(wrapper)
