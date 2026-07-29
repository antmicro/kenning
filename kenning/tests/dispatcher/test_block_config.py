from kenning.dispatcher.block_config import (
    BLOCK_CONFIG_PARAMETERS_KEY,
    BLOCK_CONFIGURATIONS_KEY,
    BLOCK_DIRECT_ARGUMENTS_KEY,
    UNAFFILIATED_PARAMETERS_KEY,
    ConfigKey,
    apply_default_blocks_by_block_type,
    argparse_to_config_dict,
    filter_block_types_from_config_dict,
    merge_config_dicts,
    set_block_direct_argument,
    yaml_or_json_to_config_dict,
)


class TestBlockConfig:
    def test_argparse_to_config_dict(self):
        class MockArgparse:
            def __init__(self):
                self.modelwrapper_cls = "ExampleClassOne"
                self.dataset_cls = "ExampleClassTwo"
                self.runtime_cls = "ExampleClassThree"
                self.platform_cls = "ExampleClassFour"
                self.param_one = 34
                self.param_two = "lorem ipsum"
                self.param_three = [45.4, 1.9]
                self.param_four = True

        args = MockArgparse()
        empty_block_dict = {
            BLOCK_CONFIG_PARAMETERS_KEY: {},
            BLOCK_DIRECT_ARGUMENTS_KEY: {},
        }
        desired_dict = {
            BLOCK_CONFIGURATIONS_KEY: {
                ConfigKey.model_wrapper: {
                    args.modelwrapper_cls: empty_block_dict
                },
                ConfigKey.dataset: {args.dataset_cls: empty_block_dict},
                ConfigKey.runtime: {args.runtime_cls: empty_block_dict},
                ConfigKey.platform: {args.platform_cls: empty_block_dict},
            },
            UNAFFILIATED_PARAMETERS_KEY: {
                "param_one": args.param_one,
                "param_two": args.param_two,
                "param_three": args.param_three,
                "param_four": args.param_four,
            },
        }
        assert desired_dict == argparse_to_config_dict(args)

    def test_yaml_or_json_to_config_dict(self):
        mock_json_config = {
            "model_wrapper": {
                "type": "PyTorchPetDatasetMobileNetV2",
                "parameters": {
                    "model_path": "kenning:///models/classification/pytorch_pet_dataset_mobilenetv2_full_model.pth",
                    "model_name": "torch-native",
                },
            },
            "dataset": {
                "type": "PetDataset",
                "parameters": {
                    "dataset_root": "./build/PetDataset",
                    "image_memory_layout": "NCHW",
                },
            },
            "optimizers": [
                {
                    "type": "ONNXCompiler",
                    "parameters": {
                        "compiled_model_path": "./build/optimized_mobilenetv2.onnx",  # noqa: E501
                    },
                },
                {
                    "type": "ExecuTorchOptimizer",
                    "parameters": {
                        "compiled_model_path": "./build/optimized_mobilenetv2.pte",  # noqa: E501
                        "quantize": True,
                        "backends": ["XNNPACK"],
                    },
                },
            ],
            "runtime": {
                "type": "ExecuTorchRuntime",
                "parameters": {
                    "save_model_path": "./build/optimized_mobilenetv2.pte",
                    "image_memory_layout": "NCHW",
                    "measurements": ["path/to/file"],
                },
            },
            "report": {"parameters": {"measurements": ["path/to/file"]}},
        }
        desired_dict = {
            BLOCK_CONFIGURATIONS_KEY: {
                ConfigKey.model_wrapper: {
                    "PyTorchPetDatasetMobileNetV2": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "model_path": "kenning:///models/classification/pytorch_pet_dataset_mobilenetv2_full_model.pth",
                            "model_name": "torch-native",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                },
                ConfigKey.dataset: {
                    "PetDataset": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "dataset_root": "./build/PetDataset",
                            "image_memory_layout": "NCHW",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                },
                ConfigKey.optimizers: {
                    "ONNXCompiler": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "compiled_model_path": "./build/optimized_mobilenetv2.onnx",  # noqa: E501
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                    "ExecuTorchOptimizer": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "compiled_model_path": "./build/optimized_mobilenetv2.pte",  # noqa: E501
                            "quantize": True,
                            "backends": ["XNNPACK"],
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                },
                ConfigKey.runtime: {
                    "ExecuTorchRuntime": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "save_model_path": "./build/optimized_mobilenetv2.pte",  # noqa: E501
                            "image_memory_layout": "NCHW",
                            "measurements": ["path/to/file"],
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                },
                ConfigKey.report: {
                    None: {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "measurements": ["path/to/file"]
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    }
                },
            },
            UNAFFILIATED_PARAMETERS_KEY: {},
        }
        assert desired_dict == yaml_or_json_to_config_dict(mock_json_config)

    def test_merge_config_dicts(self):
        dict_one = {
            BLOCK_CONFIGURATIONS_KEY: {
                ConfigKey.model_wrapper: {
                    "PyTorchPetDatasetMobileNetV2": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param1": "lorem ipsum",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "a": "b",
                            "argument": [5.98],
                        },
                    },
                },
                ConfigKey.dataset: {
                    "TabularDataset": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param1": 56,
                            "param2": 98,
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "argument": ["dfdsfsd", "fdf"],
                        },
                    },
                },
                ConfigKey.optimizers: {
                    "ONNXCompiler": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param1": 325,
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                },
                ConfigKey.report: {
                    None: {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "measurements": ["path/to/file"],
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "a": 1,
                        },
                    }
                },
            },
            UNAFFILIATED_PARAMETERS_KEY: {
                "param1": 3434,
                "param2": [34, 4343, 322],
                "param3": "lorem ipsum",
            },
        }
        dict_two = {
            BLOCK_CONFIGURATIONS_KEY: {
                ConfigKey.model_wrapper: {
                    "PyTorchPetDatasetMobileNetV2": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param1": 5,
                            "param2": "lo",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                },
                ConfigKey.dataset: {
                    "TabularDataset": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param3": 98,
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "argument": ["try"],
                        },
                    },
                    "MagicWandDataset": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param9": 32,
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                },
                ConfigKey.optimizers: {
                    "ONNXCompiler": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param1": 325,
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "arg1": [1, 2, 3, 4],
                        },
                    },
                },
                ConfigKey.runtime_builder: {
                    None: {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param1": "p",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "arg1": 4,
                        },
                    },
                },
                ConfigKey.report: {
                    "MarkdownReport": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {},
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    }
                },
            },
            UNAFFILIATED_PARAMETERS_KEY: {
                "param1": 3434,
                "param3": "lorem",
                "param4": 9,
            },
        }
        expected_dict = {
            BLOCK_CONFIGURATIONS_KEY: {
                ConfigKey.model_wrapper: {
                    "PyTorchPetDatasetMobileNetV2": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param1": 5,
                            "param2": "lo",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "a": "b",
                            "argument": [5.98],
                        },
                    },
                },
                ConfigKey.dataset: {
                    "TabularDataset": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param1": 56,
                            "param2": 98,
                            "param3": 98,
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "argument": ["try"],
                        },
                    },
                    "MagicWandDataset": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param9": 32,
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                },
                ConfigKey.optimizers: {
                    "ONNXCompiler": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param1": 325,
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "arg1": [1, 2, 3, 4],
                        },
                    },
                },
                ConfigKey.runtime_builder: {
                    None: {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param1": "p",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "arg1": 4,
                        },
                    },
                },
                ConfigKey.report: {
                    None: {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "measurements": ["path/to/file"],
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "a": 1,
                        },
                    },
                    "MarkdownReport": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {},
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                },
            },
            UNAFFILIATED_PARAMETERS_KEY: {
                "param1": 3434,
                "param2": [34, 4343, 322],
                "param3": "lorem",
                "param4": 9,
            },
        }
        assert expected_dict == merge_config_dicts(dict_one, dict_two)

    def test_filter_block_types_from_config_dict(self):
        test_dict = {
            BLOCK_CONFIGURATIONS_KEY: {
                ConfigKey.model_wrapper: {
                    "PyTorchPetDatasetMobileNetV2": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {},
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                },
                ConfigKey.dataset: {
                    "TabularDataset": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {},
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                },
                ConfigKey.optimizers: {
                    "ONNXCompiler": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {},
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                },
                ConfigKey.runtime_builder: {
                    None: {
                        BLOCK_CONFIG_PARAMETERS_KEY: {},
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                },
                ConfigKey.report: {
                    None: {
                        BLOCK_CONFIG_PARAMETERS_KEY: {},
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    }
                },
            },
            UNAFFILIATED_PARAMETERS_KEY: {
                "param1": 3434,
                "param3": "lorem",
            },
        }
        expected_dict = {
            BLOCK_CONFIGURATIONS_KEY: {
                ConfigKey.optimizers: {
                    "ONNXCompiler": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {},
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                },
                ConfigKey.report: {
                    None: {
                        BLOCK_CONFIG_PARAMETERS_KEY: {},
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    }
                },
            },
            UNAFFILIATED_PARAMETERS_KEY: {
                "param1": 3434,
                "param3": "lorem",
            },
        }
        assert expected_dict == filter_block_types_from_config_dict(
            [ConfigKey.optimizers, ConfigKey.report, ConfigKey.automl],
            test_dict,
        )

    def test_apply_default_blocks_by_block_type(self):
        test_dict = {
            BLOCK_CONFIGURATIONS_KEY: {
                ConfigKey.optimizers: {
                    "ONNXCompiler": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {},
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                },
                ConfigKey.report: {
                    None: {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param1": 5,
                            "param2": "ui",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "gy": [3, 2],
                        },
                    },
                    "MarkdownReport": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "measurements": "path/to/file",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "funargument": "lorem ipsum",
                            "number": 4,
                        },
                    },
                },
            },
            UNAFFILIATED_PARAMETERS_KEY: {
                "param1": 3434,
                "param3": "lorem",
            },
        }
        test_defaults = {
            ConfigKey.optimizers: "ExecuTorchRuntime",
            ConfigKey.report: "StubReport",
            ConfigKey.model_wrapper: "GenericClassifier",
        }
        expected_dict = {
            BLOCK_CONFIGURATIONS_KEY: {
                ConfigKey.optimizers: {
                    "ONNXCompiler": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param1": 4,
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                },
                ConfigKey.report: {
                    "GenericClassifier": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param1": 5,
                            "param2": "ui",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "gy": [3, 2],
                        },
                    },
                    "MarkdownReport": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "measurements": "path/to/file",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "funargument": "lorem ipsum",
                            "number": 4,
                        },
                    },
                },
                ConfigKey.model_wrapper: {
                    "GenericClassifier": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {},
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                },
            },
            UNAFFILIATED_PARAMETERS_KEY: {
                "param1": 3434,
                "param3": "lorem",
            },
        }
        expected_dict == apply_default_blocks_by_block_type(
            test_defaults, test_dict
        )

    def test_set_block_direct_argument(self):
        test_dict = {
            BLOCK_CONFIGURATIONS_KEY: {
                ConfigKey.optimizers: {
                    "ONNXCompiler": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {},
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                    "ExecuTorchOptimizer": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param1": 5,
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                },
                ConfigKey.report: {
                    None: {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param1": 5,
                            "param2": "ui",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "gy": [3, 2],
                        },
                    },
                    "MarkdownReport": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "measurements": "path/to/file",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "funargument": "lorem ipsum",
                            "number": 4,
                        },
                    },
                    "YetAnotherReportType": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "measurements": "path/to/file",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "number": 9,
                        },
                    },
                },
            },
            UNAFFILIATED_PARAMETERS_KEY: {
                "param1": 3434,
                "param3": "lorem",
            },
        }
        set_block_direct_argument("number", 99, test_dict, ConfigKey.report)
        set_block_direct_argument(
            "arg",
            ["1", "2"],
            test_dict,
            ConfigKey.optimizers,
            "ExecuTorchOptimizer",
        )
        set_block_direct_argument(
            "should not be set",
            34,
            test_dict,
            ConfigKey.optimizers,
            "TFLiteCompiler",
        )
        set_block_direct_argument(
            "should not be set", 76, test_dict, ConfigKey.model_wrapper
        )
        expected_dict = {
            BLOCK_CONFIGURATIONS_KEY: {
                ConfigKey.optimizers: {
                    "ONNXCompiler": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {},
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
                    },
                    "ExecuTorchOptimizer": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param1": 5,
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "arg": ["1", "2"],
                        },
                    },
                },
                ConfigKey.report: {
                    None: {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param1": 5,
                            "param2": "ui",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "gy": [3, 2],
                            "number": 99,
                        },
                    },
                    "MarkdownReport": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "measurements": "path/to/file",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "funargument": "lorem ipsum",
                            "number": 99,
                        },
                    },
                    "YetAnotherReportType": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "measurements": "path/to/file",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "number": 99,
                        },
                    },
                },
            },
            UNAFFILIATED_PARAMETERS_KEY: {
                "param1": 3434,
                "param3": "lorem",
            },
        }
        assert expected_dict == test_dict
