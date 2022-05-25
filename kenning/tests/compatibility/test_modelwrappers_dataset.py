import kenning.datasets.random_dataset as datasets
import kenning.modelwrappers.classification.pytorch_pet_dataset as wrappers
import kenning.resources.models
import sys
if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path


class TestModelWrapperAndDatasetCompatibility:
    def test_deliver_input(self):
        """
        Test to check if input of dataset is delivered
        to ModelWrapper
        """
        modelpath = str(path(kenning.resources.models, "classification")) +\
            "/pytorch_pet_dataset_mobilenetv2.pth"
        dataset = datasets.RandomizedClassificationDataset("")
        wrappers.PyTorchPetDatasetMobileNetV2(
            modelpath,
            dataset
        )

    def test_without_dataset(self):
        """
        Perform tests on ModelWrapper with provided dataset as None
        """
        modelpath = str(path(kenning.resources.models, "classification")) +\
            "/pytorch_pet_dataset_mobilenetv2.pth"
        wrappers.PyTorchPetDatasetMobileNetV2(    # noqa: E501
            modelpath,
            None
        )
