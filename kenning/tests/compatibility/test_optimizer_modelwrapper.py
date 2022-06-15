class TestOptimizerModelWrapper:
    def test_compile_existance_models(self, optimizerSamples,
                                      modelSamples, modelwrapperSamples):
        """
        Tests compilation process for models presented in kenning.

        List of methods are being tested
        --------------------------------
        Optimizer.consult_model_type()
        Optimizer.compile()

        Used fixtures
        -------------
        optimizerSamples - to get optimizer instances.
        modelSamples - to get pathes for models to compile.
        modelwrapperSamples - to get inputshape and data type.
        """

        def run_tests(optimizer, model_path, wrapper_name, **kwargs):
            """
            Arguments
            ---------
            optimizer: Optimizer
                The optimizer instance that being tested.
            model_path: str
                The path to input model.
            wrapper_name: str
                The name of modelwrapper that is compatible with input model.
            """

            wrapper = modelwrapperSamples.get(wrapper_name)
            inputshapes, dtype = wrapper.get_input_spec()
            optimizer.compile(model_path, inputshapes, dtype=dtype)

        for optimizer, modelframework in optimizerSamples:
            run_tests(optimizer, *modelSamples.get(modelframework))

    def test_onnx_model_optimization(self, modelwrapperSamples):
        """
        Tests saving models to onnx format with modelwrappers
        and converting them using optimizers.

        List of methods are being tested
        --------------------------------
        ModelWrapper.save_to_onnx()
        Optimizer.consult_model_type()

        Used fixtures
        -------------
        modelwrapperSamples - to get modelwrappers instances.
        """

        def save_to_onnx(warpper):
            print(wrapper)

        for wrapper in modelwrapperSamples:
            save_to_onnx(wrapper)
