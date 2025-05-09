# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from kenning.tests.conftest import DataFolder, Samples


class TestModelWrapperAndDatasetCompatibility:
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "datasetname",
        [
            ("PetDataset"),
        ],
    )
    def test_deliver_input(self, datasetsamples: Samples, datasetname):
        """
        Tests dataset functions used by modelwrappers for data delivering.

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
        dataset = datasetsamples.get(datasetname)

        Xt, Xv, Yt, Yv = dataset.train_test_split_representations(
            test_fraction=0.25
        )
        assert isinstance(Xt, list) and isinstance(Xv, list)
        assert len(Xt) > 0 and len(Xv) > 0
        assert len(Xt) == len(Yt) and len(Xv) == len(Yv)

        samples = dataset.prepare_input_samples(Xv)
        assert isinstance(samples, list)
        assert all(len(s) == len(Xv) for s in samples)

        samples = dataset.prepare_output_samples(Yv)
        assert isinstance(samples, list)
        assert all(len(s) == len(Yv) for s in samples)

        mean_and_std = dataset.get_input_mean_std()
        assert isinstance(mean_and_std, tuple)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "wrappername",
        [
            ("PyTorchPetDatasetMobileNetV2"),
            ("TensorFlowPetDatasetMobileNetV2"),
        ],
    )
    def test_deliver_output(
        self,
        wrappername: str,
        datasetimages: DataFolder,
        modelwrappersamples: Samples,
    ):
        """
        Tests modelwrapper functions to deliver output to datasets.

        List of methods that are being tested
        --------------------------------
        modelwrapper.test_inference()
        dataset.evaluate()

        Used fixtures
        -------------
        datasetimages - to get total amount of images.
        modelwrappersamples - to get modelwrapper instances.
        """
        wrapper = modelwrappersamples.get(wrappername)
        from kenning.core.measurements import Measurements

        measurements = wrapper.test_inference()
        assert isinstance(measurements, Measurements)
        assert measurements.get_values("total") == int(
            datasetimages.amount * wrapper.dataset.split_fraction_test
        )
