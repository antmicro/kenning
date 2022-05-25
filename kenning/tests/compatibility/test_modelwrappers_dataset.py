import kenning.datasets.random_dataset as datasets
import kenning.modelwrappers.classification.pytorch_pet_dataset as wrappers
import kenning.resources.models
import sys
if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path


class TestModelWrapperAndDatasetCompatibility:
    def test_deliver_input(self, empty_dir):
        """
        Test to check if input of dataset is delivered
        to ModelWrapper
        """
        modelpath = str(path(kenning.resources.models, "classification")) +\
            "/pytorch_pet_dataset_mobilenetv2.pth"
        dataset = datasets.RandomizedClassificationDataset(empty_dir)    # noqa: E501
        wrappers.PyTorchPetDatasetMobileNetV2(       # noqa: E501
            modelpath,
            dataset
        )
        wrappers.PyTorchPetDatasetMobileNetV2(    # noqa: E501
            modelpath,
            None
        )
