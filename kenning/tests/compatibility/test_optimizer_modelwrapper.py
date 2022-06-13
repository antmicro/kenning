import kenning
from kenning.utils.class_loader import load_class
from pathlib import Path


class TestOptimizerModelWrapper:
    optimizer_dict = {
        'tflite_keras': (
            (
                'tflite.TFLiteCompiler',
                'pet_dataset.PetDataset',
                '/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5',    # noqa: E501
                'tflite',
                'tensorflow_pet_dataset',
            ),
            {
                'target': 'default',
                'modelframework': 'keras',
            },
        ),
        'tvm_keras': (
            (
                'tvm.TVMCompiler',
                'pet_dataset.PetDataset',
                '/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5',      # noqa: E501
                'so',
                'tensorflow_pet_dataset',
            ),
            {
                'target': 'llvm',
                'modelframework': 'keras'
            },
        ),
        'tvm_pytorch': (
            (
                'tvm.TVMCompiler',
                'pet_dataset.PetDataset',
                '/resources/models/classification/pytorch_pet_dataset_mobilenetv2_full_model.pth',      # noqa: E501
                'so',
                'pytorch_pet_dataset',
            ),
            {
                'target': 'llvm',
                'modelframework': 'torch'
            },
        ),
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

        def run_tests(images_path, optimizer_path, dataset_path, model_path,
                      file_suffix, wrapper_name, **kwargs):
            """
            Arguments
            ---------
            images_path: Path
                Path to temporary folder with images for dataset class
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
            """
            # General classes
            datasetcls = load_class("kenning.datasets." + dataset_path)
            wrappercls = load_class("kenning.modelwrappers." + self.modelwrapper_dict[wrapper_name])    # noqa: E501
            optimizercls = load_class("kenning.compilers." + optimizer_path)

            # The filename for compiled model
            file_name = optimizer_path + "_compiled." + file_suffix

            # Path where to save model
            path_to_save_model = images_path / file_name

            # Path to input model
            path_to_input_model = Path(self.kenning_path+model_path)

            # Initialize classes
            dataset = datasetcls(images_path)
            wrapper = wrappercls(images_path, dataset, from_file=False)
            optimizer = optimizercls(dataset, path_to_save_model, **kwargs)

            inputshapes, dtype = wrapper.get_input_spec()
            optimizer.compile(path_to_input_model, inputshapes, dtype=dtype)

        for optimizer_args, optimizer_kwargs in self.optimizer_dict.values():
            run_tests(fake_images.path, *optimizer_args, **optimizer_kwargs)

    def test_onnx_model_optimization(self, fake_images):
        """
        Tests saving models to onnx format by modelwrappers
        and converting them using optimizers.

        List of methods are being tested
        --------------------------------
        ModelWrapper.save_to_onnx()
        Optimizer.consult_model_type()

        Used fixtures
        -------------
        fake_images - to generate images and feed them to dataset
        """

        def save_to_onnx(warpper_path):
            wrappercls = load_class("kenning.modelwrappers." + warpper_path)
            # wrapper = wrappercls()
            print(wrappercls)

        for wrapper in self.modelwrapper_dict.values():
            save_to_onnx(wrapper)
