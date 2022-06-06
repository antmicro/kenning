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
            file_name = optimizer_path + ".compiled." + file_suffix
            path_to_save_model = (images_path / file_name).absolute()
            path_to_model = Path(self.kenning_path+model_path)
            datasetcls = load_class("kenning.datasets." + dataset_path)
            wrappercls = load_class("kenning.modelwrappers." + self.modelwrapper_dict[wrapper_name])    # noqa: E501
            optimizercls = load_class("kenning.compilers." + optimizer_path)
            dataset = datasetcls(images_path)
            wrapper = wrappercls(images_path, dataset, from_file=False)
            optimizer = optimizercls(dataset, path_to_save_model)

            inputshapes, dtype = wrapper.get_input_spec()
            print(inputshapes, dtype)
            optimizer.compile(path_to_model, inputshapes, dtype=dtype)

        for optimizer in self.optimizer_dict.values():
            run_tests(*optimizer, fake_images[0])
