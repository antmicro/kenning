import pytest
import os
import uuid
from kenning.tests.conftest import Samples
from pathlib import Path


@pytest.mark.slow
class TestOptimizerModelWrapper:
    @pytest.mark.parametrize('optimizername', [('TFLiteCompiler_keras'),
                                               ('TVMCompiler_keras'),
                                               ('TVMCompiler_torch')]
                             )
    def test_compile_existence_models(self,
                                      tmpfolder: Path,
                                      optimizername: str,
                                      modelsamples: Samples,
                                      optimizersamples: Samples,
                                      modelwrappersamples: Samples):
        """
        Tests compilation process for models presented in Kenning docs.

        List of methods that are being tested
        --------------------------------
        Optimizer.consult_model_type()
        Optimizer.compile()

        Used fixtures
        -------------
        tmpfolder - to get a folder where compiled model will be placed.
        modelsamples - to get paths for models to compile.
        optimizersamples - to get optimizer instances.
        modelwrappersamples - to get inputshape and data type.
        """

        optimizer = optimizersamples.get(optimizername)
        model_path, wrapper_name = modelsamples.get(optimizer.inputtype)
        wrapper = modelwrappersamples.get(wrapper_name)
        model_type = optimizer.consult_model_type(wrapper)
        assert isinstance(model_type, str) and len(model_type) > 0
        assert model_type in optimizer.get_input_formats()
        assert model_type in wrapper.get_output_formats()

        filepath = tmpfolder / uuid.uuid4().hex
        inputshapes, dtype = wrapper.get_input_spec()

        optimizer.set_compiled_model_path(filepath)
        optimizer.compile(model_path, inputshapes, dtype=dtype)
        assert os.path.exists(filepath)
        os.remove(filepath)

    def test_onnx_model_optimization(self,
                                     tmpfolder: Path,
                                     modelwrappersamples: Samples,
                                     optimizersamples: Samples):
        """
        Tests saving model to onnx format with modelwrappers
        and converting it using optimizers.

        List of methods that are being tested
        --------------------------------
        ModelWrapper.save_to_onnx()
        Optimizer.consult_model_type()

        Used fixtures
        -------------
        tmpfolder - to get a folder where compiled model will be placed
        optimizersamples - to get optimizers instances.
        modelwrappersamples - to get modelwrappers instances.
        """

        for wrapper_name in modelwrappersamples:
            wrapper = modelwrappersamples.get(wrapper_name)
            inputshapes, dtype = wrapper.get_input_spec()

            filename = uuid.uuid4().hex
            filepath = tmpfolder / filename
            wrapper.save_to_onnx(filepath)
            assert os.path.exists(filepath)

            for optimizer_name in optimizersamples:
                optimizer = optimizersamples.get(optimizer_name)
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
                compiled_model_path = tmpfolder / compiled_model_path
                optimizer.set_compiled_model_path(compiled_model_path)
                optimizer.compile(filepath, inputshapes, dtype=dtype)

                assert os.path.exists(compiled_model_path)
                os.remove(compiled_model_path)
            os.remove(filepath)
