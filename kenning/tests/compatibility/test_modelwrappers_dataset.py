import kenning.modelwrappers.classification.pytorch_pet_dataset as wrappers
import kenning.resources.models
import sys
if sys.version_info.minor < 9:
    from importlib_resources import path
else:
    from importlib.resources import path
from importlib import import_module


class TestModelWrapperAndDatasetCompatibility:
    def test_deliver_input(self):
        """
        Test to check if dataset's output is delivered
        to ModelWrapper. That means that we have to test datasets here.

        Test .train_test_split_representations() method.
        """

        def test_random(module_name, module_package):
            module = import_module(module_name)
            dataset = getattr(module, module_package)("")

            Xt, Xv, Yt, Yv = dataset.train_test_split_representations(0.25)

            def get_length(List):
                length = 0
                for item in List:
                    length += len(item)
                return length

            assert get_length(Xt) == get_length(Yt)
            assert get_length(Xv) == get_length(Yv)

        test_random('kenning.datasets', 'RandomizedClassificationDataset')

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
