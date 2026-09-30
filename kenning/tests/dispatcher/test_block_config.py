import pytest

from kenning.core.exceptions import ConfigurationError
from kenning.dispatcher.block_config import (
    BLOCK_CONFIG_PARAMETERS_KEY,
    BLOCK_CONFIGURATIONS_KEY,
    BLOCK_DIRECT_ARGUMENTS_KEY,
    UNAFFILIATED_PARAMETERS_KEY,
    ConfigKey,
    apply_default_blocks_by_block_type,
    argparse_to_config_dict,
    block_parameters_to_config_dict,
    filter_block_types_from_config_dict,
    get_block_direct_arguments_from_config_dict,
    get_block_parameters_from_config_dict,
    get_unaffiliated_parameter_from_config_dict,
    merge_config_dicts,
    set_block_direct_argument,
    yaml_or_json_to_config_dict,
)
from kenning.optimizers.executorch import ExecuTorchOptimizer
from kenning.optimizers.onnx import ONNXCompiler
from kenning.report.markdown_report import MarkdownReport


class TestBlockConfig:
    def test_argparse_to_config_dict(self):
        """
        Tests the function, that converts argparse.Namespace to a config dict.
        """

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
        """
        Test the function, that converts a Python dictionary, coming from a
        parsed Kenning configuration file, and converts it to a config dict.
        """
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
        """
        Tests merging of the config dicts, the dict on the right should always
        override the one on the left - recursively at every level.
        """
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
        """
        Test removal of unwanted block types from the config dict.
        """
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
        filter_block_types_from_config_dict(
            [ConfigKey.optimizers, ConfigKey.report, ConfigKey.automl],
            test_dict,
        )
        assert expected_dict == test_dict

    def test_apply_default_blocks_by_block_type(self):
        """
        Tests the function for applying defaults. If None block is in the given
        block type, the default should replace it, but adopt its config.
        """
        test_dict = {
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
                ConfigKey.dataset: {
                    "MagicWandDataset": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "path": "path/to/model/file",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "from_file": False,
                            "other_number": 4.9,
                        },
                    },
                    None: {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param5": [1, 2, 3],
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
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
            ConfigKey.dataset: "MagicWandDataset",
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
                    "StubReport": {
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
                ConfigKey.dataset: {
                    "MagicWandDataset": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "path": "path/to/model/file",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "from_file": False,
                            "other_number": 4.9,
                        },
                    },
                    None: {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param5": [1, 2, 3],
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {},
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
        apply_default_blocks_by_block_type(test_defaults, test_dict)
        assert expected_dict == test_dict

    def test_set_block_direct_argument(self):
        """
        Tests the function setting a value under the BLOCK_DIRECT_ARGUMENTS_KEY
        - both by block and by block type.
        """
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

    def test_get_unaffiliated_parameter_from_config_dict(self):
        """
        Tests retrieving free flags from the config dict, by a priority list
        (first matching name from the list gets returned).
        """
        test_dict = {
            BLOCK_CONFIGURATIONS_KEY: {},
            UNAFFILIATED_PARAMETERS_KEY: {
                "param1": 3434,
                "param2": [5.6, 9.2],
                "param3": "lorem",
            },
        }
        assert "param1", 3434 == get_unaffiliated_parameter_from_config_dict(
            test_dict, "param3", "param1"
        )

    def test_get_block_parameters_from_config_dict(self):
        """
        Tests retrieval of a specific block's user-defined parameters.
        """
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
        assert {
            "param1": 5,
            "param2": "ui",
        } == get_block_parameters_from_config_dict(None, test_dict)
        assert {
            "param1": 5,
        } == get_block_parameters_from_config_dict(
            "ExecuTorchOptimizer", test_dict, ConfigKey.optimizers
        )
        with pytest.raises(ConfigurationError):
            get_block_parameters_from_config_dict(
                "ExecuTorchOptimizer", test_dict, ConfigKey.report
            )
        with pytest.raises(ConfigurationError):
            get_block_parameters_from_config_dict(
                "nonexistantblock", test_dict
            )

    def test_get_block_direct_arguments_from_config_dict(self):
        """
        Tests retrieval of a specific block's directly passed constructor
        arguments.
        """
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
        assert {
            "gy": [3, 2],
            "number": 99,
        } == get_block_direct_arguments_from_config_dict(None, test_dict)
        assert {
            "arg": ["1", "2"],
        } == get_block_direct_arguments_from_config_dict(
            "ExecuTorchOptimizer", test_dict, ConfigKey.optimizers
        )
        with pytest.raises(ConfigurationError):
            get_block_direct_arguments_from_config_dict(
                "ExecuTorchOptimizer", test_dict, ConfigKey.report
            )
        with pytest.raises(ConfigurationError):
            get_block_direct_arguments_from_config_dict(
                "nonexistantblock", test_dict
            )

    def test_block_parameters_to_config_dict(self):
        """
        Tests creation of simple config dicts from single block configurations,
        and then subsequent merging of them, to create config for multiple
        blocks.
        """
        expected_dict = {
            BLOCK_CONFIGURATIONS_KEY: {
                ConfigKey.optimizers: {
                    "ONNXCompiler": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "param1": 2,
                        },
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
                    "MarkdownReport": {
                        BLOCK_CONFIG_PARAMETERS_KEY: {
                            "measurements": "path/to/file",
                        },
                        BLOCK_DIRECT_ARGUMENTS_KEY: {
                            "funargument": "lorem ipsum",
                            "number": 99,
                        },
                    },
                },
            },
            UNAFFILIATED_PARAMETERS_KEY: {},
        }
        dict_one = block_parameters_to_config_dict(ONNXCompiler, {"param1": 2})
        dict_two = block_parameters_to_config_dict(
            ExecuTorchOptimizer,
            {"param1": 5},
            {
                "arg": ["1", "2"],
            },
        )
        dict_three = block_parameters_to_config_dict(
            MarkdownReport,
            {
                "measurements": "path/to/file",
            },
            {
                "funargument": "lorem ipsum",
                "number": 99,
            },
        )
        final_dict = merge_config_dicts(dict_one, dict_two, dict_three)
        assert expected_dict == final_dict
