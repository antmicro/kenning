import kenning
from kenning.utils.class_loader import load_class
from pathlib import Path


class TestOptimizerModelWrapper:
    optimizer_dict = {
        'tflite': [
            'tflite.TFLiteCompiler',
            'pet_dataset.PetDataset',
            '/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5',    # noqa: E501
            'tflite',
            'tensorflow_pet_dataset',
        ],
        'tvm': [
            'tvm.TVMCompiler',
            'pet_dataset.PetDataset',
            '/resources/models/classification/pytorch_pet_dataset_mobilenetv2.pth',      # noqa: E501
            'so',
            'pytorch_pet_dataset',
        ],
    }
    modelwrapper_dict = {
        'tensorflow_pet_dataset': 'classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2',  # noqa: E501
        'pytorch_pet_dataset': 'classification.pytorch_pet_dataset.PyTorchPetDatasetMobileNetV2',           # noqa: E501
    }
    kenning_path = kenning.__path__[0]

    def test_compile_existance_models(self, fake_images):
        """
        Tests compilation process for models presented in kenning.

        List of methods are being tested
        --------------------------------
        Optimizer.consult_model_type()
        Optimizer.compile()

        Used fixtures
        -------------
        fake_images - to generate images and feed them to dataset
        """

        def run_tests(optimizer_path, dataset_path, model_path,
                      file_suffix, wrapper_name, images_path):
            """
            Arguments
            ---------
            optimizer_path: str
                An import path to optimizer
            dataset_path: str
                An import path to dataset
            model_path: str
                A path to input model
            file_suffix: str
                A suffix with which output file should be saved
            wrapper_name: str
                A key to self.modelwrapper_dict for modelwrapper to use
            images_path: pytest.fixture
                Fixture which generates a temporary folder
                with images for dataset class
            """
            # General classes
            datasetcls = load_class("kenning.datasets." + dataset_path)
            wrappercls = load_class("kenning.modelwrappers." + self.modelwrapper_dict[wrapper_name])    # noqa: E501
            optimizercls = load_class("kenning.compilers." + optimizer_path)

            # The filename for compiled model
            file_name = optimizer_path + "_compiled." + file_suffix

            # Path where to save model
            path_to_save_model = Path("/tmp/savedmodels/"+file_name)

            # Path to input model
            path_to_input_model = Path(self.kenning_path+model_path)

            # Initialize classes
            dataset = datasetcls(images_path)
            wrapper = wrappercls(path_to_input_model, dataset, from_file=True)
            optimizer = optimizercls(dataset, path_to_save_model)

            inputshapes, dtype = wrapper.get_input_spec()
            print(wrapper.get_input_spec())
            optimizer.compile(path_to_input_model, inputshapes, dtype=dtype)

        for optimizer in self.optimizer_dict.values():
            run_tests(*optimizer, fake_images[0])
