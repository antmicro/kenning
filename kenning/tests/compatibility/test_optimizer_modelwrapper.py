import pytest
import tempfile
import os


class TestOptimizerModelWrapper:
    def test_compile_existence_models(self, optimizerSamples, fake_images,
                                      modelSamples, modelwrapperSamples):
        """
        Tests compilation process for models presented in Kenning docs.

        List of methods that are being tested
        --------------------------------
        Optimizer.consult_model_type()
        Optimizer.compile()

        Used fixtures
        -------------
        optimizerSamples - to get optimizer instances.
        modelSamples - to get paths for models to compile.
        modelwrapperSamples - to get inputshape and data type.
        """

        def run_tests(optimizer, model_path, wrapper_name):
            """
            Parameters
            ---------
            optimizer: Optimizer
                The optimizer instance that being tested.
            model_path: str
                The path to input model.
            wrapper_name: str
                The name of modelwrapper that is compatible with input model.
            """

            wrapper = modelwrapperSamples.get(wrapper_name)
            model_type = optimizer.consult_model_type(wrapper)
            assert isinstance(model_type, str) and len(model_type) > 0
            assert model_type in optimizer.get_input_formats()
            assert model_type in wrapper.get_output_formats()

            filepath = tempfile.NamedTemporaryFile().name[5:]
            filepath = fake_images.path / filepath
            inputshapes, dtype = wrapper.get_input_spec()

            optimizer.set_compiled_model_path(filepath)
            optimizer.compile(model_path, inputshapes, dtype=dtype)
            assert os.path.exists(filepath)
            os.remove(filepath)

        for optimizer in optimizerSamples:
            run_tests(optimizer, *modelSamples.get(optimizer.inputtype))

    def test_onnx_model_optimization(self, modelwrapperSamples,
                                     optimizerSamples, fake_images):
        """
        Tests saving model to onnx format with modelwrappers
        and converting it using optimizers.

        List of methods that are being tested
        --------------------------------
        ModelWrapper.save_to_onnx()
        Optimizer.consult_model_type()

        Used fixtures
        -------------
        modelwrapperSamples - to get modelwrappers instances.
        optimizerSamples - to get optimizers instances.
        """

        for wrapper in modelwrapperSamples:
            inputshapes, dtype = wrapper.get_input_spec()

            filename = tempfile.NamedTemporaryFile().name[5:]
            filepath = fake_images.path / filename
            wrapper.save_to_onnx(filepath)
            assert os.path.exists(filepath)

            for optimizer in optimizerSamples:
                with pytest.raises(ValueError):
                    optimizer.consult_model_type(optimizer)
                # TODO: In future there might be no shared model types,
                # so method may throw an exception
                model_type = optimizer.consult_model_type(wrapper)
                assert isinstance(model_type, str)
                assert model_type in optimizer.get_input_formats()
                assert model_type in wrapper.get_output_formats()

                optimizer.set_input_type('onnx')
                compiled_model_path = (filename + '_' + optimizer.inputtype)
                compiled_model_path = fake_images.path / compiled_model_path
                optimizer.set_compiled_model_path(compiled_model_path)
                optimizer.compile(filepath, inputshapes, dtype=dtype)

                assert os.path.exists(compiled_model_path)
                os.remove(compiled_model_path)

            os.remove(filepath)
