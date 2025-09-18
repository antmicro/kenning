
## Comparison of workspace.automl-results.1234_3_5.0.measurements.json, workspace.automl-results.1234_12_5.0.measurements.json, workspace.automl-results.1234_21_5.0.measurements.json, workspace.automl-results.1234_29_1.6666666666666665.measurements.json and workspace.automl-results.1234_29_5.0.measurements.json


### Commands used

````{note}

This section was generated using:

```bash
python -m kenning.__main__ \
    automl \
    optimize \
    test \
    report \
    --cfg \
        /home/runner/work/kenning/kenning/scripts/configs/automl-scenario.yml \
    --report-path \
        /home/runner/work/kenning/kenning/docs/source/generated/sample-automl.md \
    --allow-failures \
    --to-html \
    --ver \
        DEBUG \
    --root-dir \
        /home/runner/work/kenning/kenning/docs/source \
    --img-dir \
        /home/runner/work/kenning/kenning/docs/source/generated/img \
    --comparison-only \
    --smaller-header \
    --main-quality-metric \
        f1

python -m kenning.__main__ \
    automl \
    optimize \
    test \
    report \
    --cfg \
        /home/runner/work/kenning/kenning/scripts/configs/automl-scenario.yml \
    --report-path \
        /home/runner/work/kenning/kenning/docs/source/generated/sample-automl.md \
    --allow-failures \
    --to-html \
    --ver \
        DEBUG \
    --root-dir \
        /home/runner/work/kenning/kenning/docs/source \
    --img-dir \
        /home/runner/work/kenning/kenning/docs/source/generated/img \
    --comparison-only \
    --smaller-header \
    --main-quality-metric \
        f1

python -m kenning.__main__ \
    automl \
    optimize \
    test \
    report \
    --cfg \
        /home/runner/work/kenning/kenning/scripts/configs/automl-scenario.yml \
    --report-path \
        /home/runner/work/kenning/kenning/docs/source/generated/sample-automl.md \
    --allow-failures \
    --to-html \
    --ver \
        DEBUG \
    --root-dir \
        /home/runner/work/kenning/kenning/docs/source \
    --img-dir \
        /home/runner/work/kenning/kenning/docs/source/generated/img \
    --comparison-only \
    --smaller-header \
    --main-quality-metric \
        f1

python -m kenning.__main__ \
    automl \
    optimize \
    test \
    report \
    --cfg \
        /home/runner/work/kenning/kenning/scripts/configs/automl-scenario.yml \
    --report-path \
        /home/runner/work/kenning/kenning/docs/source/generated/sample-automl.md \
    --allow-failures \
    --to-html \
    --ver \
        DEBUG \
    --root-dir \
        /home/runner/work/kenning/kenning/docs/source \
    --img-dir \
        /home/runner/work/kenning/kenning/docs/source/generated/img \
    --comparison-only \
    --smaller-header \
    --main-quality-metric \
        f1

python -m kenning.__main__ \
    automl \
    optimize \
    test \
    report \
    --cfg \
        /home/runner/work/kenning/kenning/scripts/configs/automl-scenario.yml \
    --report-path \
        /home/runner/work/kenning/kenning/docs/source/generated/sample-automl.md \
    --allow-failures \
    --to-html \
    --ver \
        DEBUG \
    --root-dir \
        /home/runner/work/kenning/kenning/docs/source \
    --img-dir \
        /home/runner/work/kenning/kenning/docs/source/generated/img \
    --comparison-only \
    --smaller-header \
    --main-quality-metric \
        f1


```
````


### General information for workspace.automl-results.1234_3_5.0.measurements.json

*Model framework*:

* torch ver. 2.3.1+cu121

*Input JSON*:

```json
{
    "dataset": {
        "type": "kenning.datasets.anomaly_detection_dataset.AnomalyDetectionDataset",
        "parameters": {
            "csv_file": "kenning:///datasets/anomaly_detection/cats_nano.csv",
            "window_size": 5,
            "gather_predictions": true,
            "dataset_root": "workspace/CATS",
            "inference_batch_size": 1,
            "download_dataset": true,
            "force_download_dataset": false,
            "external_calibration_dataset": null,
            "split_fraction_test": 0.1,
            "split_fraction_val": null,
            "split_seed": 12345,
            "reduce_dataset": 1.0
        }
    },
    "dataconverter": {
        "type": "kenning.dataconverters.modelwrapper_dataconverter.ModelWrapperDataConverter",
        "parameters": {}
    },
    "optimizers": [
        {
            "type": "kenning.optimizers.tflite.TFLiteCompiler",
            "parameters": {
                "model_framework": "any",
                "target": "default",
                "inference_input_type": "float32",
                "inference_output_type": "float32",
                "dataset_percentage": 0.25,
                "quantization_aware_training": false,
                "use_tf_select_ops": false,
                "resolver_template_path": null,
                "resolver_output_path": null,
                "epochs": 3,
                "batch_size": 32,
                "optimizer": "adam",
                "disable_from_logits": false,
                "save_to_zip": false,
                "compiled_model_path": "workspace/automl-results/vae.0.tflite",
                "location": "host"
            }
        }
    ],
    "platform": {
        "type": "kenning.platforms.zephyr.ZephyrPlatform",
        "parameters": {
            "zephyr_build_path": "workspace/kzr_build",
            "llext_binary_path": null,
            "uart_port": "/tmp/renode_uart_g1rgi17k/uart",
            "uart_baudrate": 115200,
            "uart_log_port": "/tmp/renode_uart_g1rgi17k/uart_log",
            "uart_log_baudrate": 115200,
            "auto_flash": false,
            "openocd_path": "openocd",
            "sensor": null,
            "number_of_batches": 16,
            "simulated": true,
            "runtime_binary_path": null,
            "platform_resc_path": "gh://antmicro:kenning-zephyr-runtime/renode/scripts/max32690evkit.resc;branch=main",
            "resc_dependencies": [],
            "post_start_commands": [],
            "disable_opcode_counters": false,
            "disable_profiler": false,
            "profiler_dump_path": "/tmp/renode_profiler_mxb0_nox.dump",
            "profiler_interval_step": 10.0,
            "runtime_init_log_msg": "Inference server started",
            "runtime_init_timeout": 30,
            "name": "max32690evkit/max32690/m4",
            "platforms_definitions": [
                "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/.venv/lib/python3.11/site-packages/kenning/resources/platforms/platforms.yml"
            ]
        }
    },
    "protocol": {
        "type": "kenning.protocols.uart.UARTProtocol",
        "parameters": {
            "port": "/tmp/renode_uart_g1rgi17k/uart",
            "baudrate": 115200,
            "error_recovery": true,
            "timeout": 30
        }
    },
    "model_wrapper": {
        "type": "kenning.modelwrappers.anomaly_detection.vae.PyTorchAnomalyDetectionVAE",
        "parameters": {
            "encoder_neuron_list": [
                16,
                8
            ],
            "decoder_neuron_list": [
                16,
                32
            ],
            "latent_dim": 2,
            "hidden_activation": "relu",
            "output_activation": "sigmoid",
            "batch_norm": false,
            "dropout_rate": 0.0,
            "loss_beta": 1.0,
            "loss_capacity": 0.0,
            "clip_grad_max_norm": 2.0,
            "batch_size": null,
            "learning_rate": null,
            "num_epochs": null,
            "evaluate": true,
            "model_path": "workspace/automl-results/1234_3_5.0.pth",
            "model_name": null
        }
    },
    "runtime": {
        "type": "kenning.runtimes.tflite.TFLiteRuntime",
        "parameters": {
            "save_model_path": "workspace/automl-results/vae.0.tflite",
            "delegates_list": null,
            "num_threads": 4,
            "llext_binary_path": null,
            "disable_performance_measurements": false
        }
    },
    "runtime_builder": {
        "type": "kenning.runtimebuilders.zephyr.ZephyrRuntimeBuilder",
        "parameters": {
            "board": "max32690evkit/max32690/m4",
            "application_dir": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/app",
            "build_dir": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/build",
            "venv_dir": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/.west-venv",
            "extra_targets": [
                "board-repl"
            ],
            "extra_build_args": [],
            "use_llext": false,
            "workspace": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime",
            "output_path": "workspace/kzr_build",
            "model_framework": "tflite"
        }
    }
}

```


### General information for workspace.automl-results.1234_12_5.0.measurements.json

*Model framework*:

* torch ver. 2.3.1+cu121

*Input JSON*:

```json
{
    "dataset": {
        "type": "kenning.datasets.anomaly_detection_dataset.AnomalyDetectionDataset",
        "parameters": {
            "csv_file": "kenning:///datasets/anomaly_detection/cats_nano.csv",
            "window_size": 5,
            "gather_predictions": true,
            "dataset_root": "workspace/CATS",
            "inference_batch_size": 1,
            "download_dataset": true,
            "force_download_dataset": false,
            "external_calibration_dataset": null,
            "split_fraction_test": 0.1,
            "split_fraction_val": null,
            "split_seed": 12345,
            "reduce_dataset": 1.0
        }
    },
    "dataconverter": {
        "type": "kenning.dataconverters.modelwrapper_dataconverter.ModelWrapperDataConverter",
        "parameters": {}
    },
    "optimizers": [
        {
            "type": "kenning.optimizers.tflite.TFLiteCompiler",
            "parameters": {
                "model_framework": "any",
                "target": "default",
                "inference_input_type": "float32",
                "inference_output_type": "float32",
                "dataset_percentage": 0.25,
                "quantization_aware_training": false,
                "use_tf_select_ops": false,
                "resolver_template_path": null,
                "resolver_output_path": null,
                "epochs": 3,
                "batch_size": 32,
                "optimizer": "adam",
                "disable_from_logits": false,
                "save_to_zip": false,
                "compiled_model_path": "workspace/automl-results/vae.1.tflite",
                "location": "host"
            }
        }
    ],
    "platform": {
        "type": "kenning.platforms.zephyr.ZephyrPlatform",
        "parameters": {
            "zephyr_build_path": "workspace/kzr_build",
            "llext_binary_path": null,
            "uart_port": "/tmp/renode_uart_oxp8vwer/uart",
            "uart_baudrate": 115200,
            "uart_log_port": "/tmp/renode_uart_oxp8vwer/uart_log",
            "uart_log_baudrate": 115200,
            "auto_flash": false,
            "openocd_path": "openocd",
            "sensor": null,
            "number_of_batches": 16,
            "simulated": true,
            "runtime_binary_path": null,
            "platform_resc_path": "gh://antmicro:kenning-zephyr-runtime/renode/scripts/max32690evkit.resc;branch=main",
            "resc_dependencies": [],
            "post_start_commands": [],
            "disable_opcode_counters": false,
            "disable_profiler": false,
            "profiler_dump_path": "/tmp/renode_profiler_94hvoct2.dump",
            "profiler_interval_step": 10.0,
            "runtime_init_log_msg": "Inference server started",
            "runtime_init_timeout": 30,
            "name": "max32690evkit/max32690/m4",
            "platforms_definitions": [
                "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/.venv/lib/python3.11/site-packages/kenning/resources/platforms/platforms.yml"
            ]
        }
    },
    "protocol": {
        "type": "kenning.protocols.uart.UARTProtocol",
        "parameters": {
            "port": "/tmp/renode_uart_oxp8vwer/uart",
            "baudrate": 115200,
            "error_recovery": true,
            "timeout": 30
        }
    },
    "model_wrapper": {
        "type": "kenning.modelwrappers.anomaly_detection.vae.PyTorchAnomalyDetectionVAE",
        "parameters": {
            "encoder_neuron_list": [
                14,
                4,
                22,
                47
            ],
            "decoder_neuron_list": [
                10,
                17,
                33,
                12,
                19
            ],
            "latent_dim": 28,
            "hidden_activation": "tanh",
            "output_activation": "tanh",
            "batch_norm": true,
            "dropout_rate": 0.2675842741318,
            "loss_beta": 0.5682944824213,
            "loss_capacity": 0.097722158423,
            "clip_grad_max_norm": 2.836312616794,
            "batch_size": null,
            "learning_rate": null,
            "num_epochs": null,
            "evaluate": true,
            "model_path": "workspace/automl-results/1234_12_5.0.pth",
            "model_name": null
        }
    },
    "runtime": {
        "type": "kenning.runtimes.tflite.TFLiteRuntime",
        "parameters": {
            "save_model_path": "workspace/automl-results/vae.1.tflite",
            "delegates_list": null,
            "num_threads": 4,
            "llext_binary_path": null,
            "disable_performance_measurements": false
        }
    },
    "runtime_builder": {
        "type": "kenning.runtimebuilders.zephyr.ZephyrRuntimeBuilder",
        "parameters": {
            "board": "max32690evkit/max32690/m4",
            "application_dir": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/app",
            "build_dir": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/build",
            "venv_dir": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/.west-venv",
            "extra_targets": [
                "board-repl"
            ],
            "extra_build_args": [],
            "use_llext": false,
            "workspace": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime",
            "output_path": "workspace/kzr_build",
            "model_framework": "tflite"
        }
    }
}

```


### General information for workspace.automl-results.1234_21_5.0.measurements.json

*Model framework*:

* torch ver. 2.3.1+cu121

*Input JSON*:

```json
{
    "dataset": {
        "type": "kenning.datasets.anomaly_detection_dataset.AnomalyDetectionDataset",
        "parameters": {
            "csv_file": "kenning:///datasets/anomaly_detection/cats_nano.csv",
            "window_size": 5,
            "gather_predictions": true,
            "dataset_root": "workspace/CATS",
            "inference_batch_size": 1,
            "download_dataset": true,
            "force_download_dataset": false,
            "external_calibration_dataset": null,
            "split_fraction_test": 0.1,
            "split_fraction_val": null,
            "split_seed": 12345,
            "reduce_dataset": 1.0
        }
    },
    "dataconverter": {
        "type": "kenning.dataconverters.modelwrapper_dataconverter.ModelWrapperDataConverter",
        "parameters": {}
    },
    "optimizers": [
        {
            "type": "kenning.optimizers.tflite.TFLiteCompiler",
            "parameters": {
                "model_framework": "any",
                "target": "default",
                "inference_input_type": "float32",
                "inference_output_type": "float32",
                "dataset_percentage": 0.25,
                "quantization_aware_training": false,
                "use_tf_select_ops": false,
                "resolver_template_path": null,
                "resolver_output_path": null,
                "epochs": 3,
                "batch_size": 32,
                "optimizer": "adam",
                "disable_from_logits": false,
                "save_to_zip": false,
                "compiled_model_path": "workspace/automl-results/vae.2.tflite",
                "location": "host"
            }
        }
    ],
    "platform": {
        "type": "kenning.platforms.zephyr.ZephyrPlatform",
        "parameters": {
            "zephyr_build_path": "workspace/kzr_build",
            "llext_binary_path": null,
            "uart_port": "/tmp/renode_uart_zqw3t5u0/uart",
            "uart_baudrate": 115200,
            "uart_log_port": "/tmp/renode_uart_zqw3t5u0/uart_log",
            "uart_log_baudrate": 115200,
            "auto_flash": false,
            "openocd_path": "openocd",
            "sensor": null,
            "number_of_batches": 16,
            "simulated": true,
            "runtime_binary_path": null,
            "platform_resc_path": "gh://antmicro:kenning-zephyr-runtime/renode/scripts/max32690evkit.resc;branch=main",
            "resc_dependencies": [],
            "post_start_commands": [],
            "disable_opcode_counters": false,
            "disable_profiler": false,
            "profiler_dump_path": "/tmp/renode_profiler_lzmvee3p.dump",
            "profiler_interval_step": 10.0,
            "runtime_init_log_msg": "Inference server started",
            "runtime_init_timeout": 30,
            "name": "max32690evkit/max32690/m4",
            "platforms_definitions": [
                "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/.venv/lib/python3.11/site-packages/kenning/resources/platforms/platforms.yml"
            ]
        }
    },
    "protocol": {
        "type": "kenning.protocols.uart.UARTProtocol",
        "parameters": {
            "port": "/tmp/renode_uart_zqw3t5u0/uart",
            "baudrate": 115200,
            "error_recovery": true,
            "timeout": 30
        }
    },
    "model_wrapper": {
        "type": "kenning.modelwrappers.anomaly_detection.vae.PyTorchAnomalyDetectionVAE",
        "parameters": {
            "encoder_neuron_list": [
                27,
                36,
                15
            ],
            "decoder_neuron_list": [
                34,
                19
            ],
            "latent_dim": 38,
            "hidden_activation": "relu",
            "output_activation": "sigmoid",
            "batch_norm": true,
            "dropout_rate": 0.2108925288844,
            "loss_beta": 0.3151576483133,
            "loss_capacity": 0.164325661097,
            "clip_grad_max_norm": 5.1411999068705,
            "batch_size": null,
            "learning_rate": null,
            "num_epochs": null,
            "evaluate": true,
            "model_path": "workspace/automl-results/1234_21_5.0.pth",
            "model_name": null
        }
    },
    "runtime": {
        "type": "kenning.runtimes.tflite.TFLiteRuntime",
        "parameters": {
            "save_model_path": "workspace/automl-results/vae.2.tflite",
            "delegates_list": null,
            "num_threads": 4,
            "llext_binary_path": null,
            "disable_performance_measurements": false
        }
    },
    "runtime_builder": {
        "type": "kenning.runtimebuilders.zephyr.ZephyrRuntimeBuilder",
        "parameters": {
            "board": "max32690evkit/max32690/m4",
            "application_dir": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/app",
            "build_dir": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/build",
            "venv_dir": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/.west-venv",
            "extra_targets": [
                "board-repl"
            ],
            "extra_build_args": [],
            "use_llext": false,
            "workspace": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime",
            "output_path": "workspace/kzr_build",
            "model_framework": "tflite"
        }
    }
}

```


### General information for workspace.automl-results.1234_29_1.6666666666666665.measurements.json

*Model framework*:

* torch ver. 2.3.1+cu121

*Input JSON*:

```json
{
    "dataset": {
        "type": "kenning.datasets.anomaly_detection_dataset.AnomalyDetectionDataset",
        "parameters": {
            "csv_file": "kenning:///datasets/anomaly_detection/cats_nano.csv",
            "window_size": 5,
            "gather_predictions": true,
            "dataset_root": "workspace/CATS",
            "inference_batch_size": 1,
            "download_dataset": true,
            "force_download_dataset": false,
            "external_calibration_dataset": null,
            "split_fraction_test": 0.1,
            "split_fraction_val": null,
            "split_seed": 12345,
            "reduce_dataset": 1.0
        }
    },
    "dataconverter": {
        "type": "kenning.dataconverters.modelwrapper_dataconverter.ModelWrapperDataConverter",
        "parameters": {}
    },
    "optimizers": [
        {
            "type": "kenning.optimizers.tflite.TFLiteCompiler",
            "parameters": {
                "model_framework": "any",
                "target": "default",
                "inference_input_type": "float32",
                "inference_output_type": "float32",
                "dataset_percentage": 0.25,
                "quantization_aware_training": false,
                "use_tf_select_ops": false,
                "resolver_template_path": null,
                "resolver_output_path": null,
                "epochs": 3,
                "batch_size": 32,
                "optimizer": "adam",
                "disable_from_logits": false,
                "save_to_zip": false,
                "compiled_model_path": "workspace/automl-results/vae.3.tflite",
                "location": "host"
            }
        }
    ],
    "platform": {
        "type": "kenning.platforms.zephyr.ZephyrPlatform",
        "parameters": {
            "zephyr_build_path": "workspace/kzr_build",
            "llext_binary_path": null,
            "uart_port": "/tmp/renode_uart_jcxt6nck/uart",
            "uart_baudrate": 115200,
            "uart_log_port": "/tmp/renode_uart_jcxt6nck/uart_log",
            "uart_log_baudrate": 115200,
            "auto_flash": false,
            "openocd_path": "openocd",
            "sensor": null,
            "number_of_batches": 16,
            "simulated": true,
            "runtime_binary_path": null,
            "platform_resc_path": "gh://antmicro:kenning-zephyr-runtime/renode/scripts/max32690evkit.resc;branch=main",
            "resc_dependencies": [],
            "post_start_commands": [],
            "disable_opcode_counters": false,
            "disable_profiler": false,
            "profiler_dump_path": "/tmp/renode_profiler_t9ere011.dump",
            "profiler_interval_step": 10.0,
            "runtime_init_log_msg": "Inference server started",
            "runtime_init_timeout": 30,
            "name": "max32690evkit/max32690/m4",
            "platforms_definitions": [
                "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/.venv/lib/python3.11/site-packages/kenning/resources/platforms/platforms.yml"
            ]
        }
    },
    "protocol": {
        "type": "kenning.protocols.uart.UARTProtocol",
        "parameters": {
            "port": "/tmp/renode_uart_jcxt6nck/uart",
            "baudrate": 115200,
            "error_recovery": true,
            "timeout": 30
        }
    },
    "model_wrapper": {
        "type": "kenning.modelwrappers.anomaly_detection.vae.PyTorchAnomalyDetectionVAE",
        "parameters": {
            "encoder_neuron_list": [
                34,
                8,
                14,
                40,
                33
            ],
            "decoder_neuron_list": [
                44,
                15,
                36,
                25,
                16
            ],
            "latent_dim": 12,
            "hidden_activation": "softmax",
            "output_activation": "tanh",
            "batch_norm": true,
            "dropout_rate": 0.4054920288479,
            "loss_beta": 0.4235748410695,
            "loss_capacity": 0.7909717876419,
            "clip_grad_max_norm": 8.5833348167468,
            "batch_size": null,
            "learning_rate": null,
            "num_epochs": null,
            "evaluate": true,
            "model_path": "workspace/automl-results/1234_29_1.6666666666666665.pth",
            "model_name": null
        }
    },
    "runtime": {
        "type": "kenning.runtimes.tflite.TFLiteRuntime",
        "parameters": {
            "save_model_path": "workspace/automl-results/vae.3.tflite",
            "delegates_list": null,
            "num_threads": 4,
            "llext_binary_path": null,
            "disable_performance_measurements": false
        }
    },
    "runtime_builder": {
        "type": "kenning.runtimebuilders.zephyr.ZephyrRuntimeBuilder",
        "parameters": {
            "board": "max32690evkit/max32690/m4",
            "application_dir": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/app",
            "build_dir": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/build",
            "venv_dir": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/.west-venv",
            "extra_targets": [
                "board-repl"
            ],
            "extra_build_args": [],
            "use_llext": false,
            "workspace": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime",
            "output_path": "workspace/kzr_build",
            "model_framework": "tflite"
        }
    }
}

```


### General information for workspace.automl-results.1234_29_5.0.measurements.json

*Model framework*:

* torch ver. 2.3.1+cu121

*Input JSON*:

```json
{
    "dataset": {
        "type": "kenning.datasets.anomaly_detection_dataset.AnomalyDetectionDataset",
        "parameters": {
            "csv_file": "kenning:///datasets/anomaly_detection/cats_nano.csv",
            "window_size": 5,
            "gather_predictions": true,
            "dataset_root": "workspace/CATS",
            "inference_batch_size": 1,
            "download_dataset": true,
            "force_download_dataset": false,
            "external_calibration_dataset": null,
            "split_fraction_test": 0.1,
            "split_fraction_val": null,
            "split_seed": 12345,
            "reduce_dataset": 1.0
        }
    },
    "dataconverter": {
        "type": "kenning.dataconverters.modelwrapper_dataconverter.ModelWrapperDataConverter",
        "parameters": {}
    },
    "optimizers": [
        {
            "type": "kenning.optimizers.tflite.TFLiteCompiler",
            "parameters": {
                "model_framework": "any",
                "target": "default",
                "inference_input_type": "float32",
                "inference_output_type": "float32",
                "dataset_percentage": 0.25,
                "quantization_aware_training": false,
                "use_tf_select_ops": false,
                "resolver_template_path": null,
                "resolver_output_path": null,
                "epochs": 3,
                "batch_size": 32,
                "optimizer": "adam",
                "disable_from_logits": false,
                "save_to_zip": false,
                "compiled_model_path": "workspace/automl-results/vae.4.tflite",
                "location": "host"
            }
        }
    ],
    "platform": {
        "type": "kenning.platforms.zephyr.ZephyrPlatform",
        "parameters": {
            "zephyr_build_path": "workspace/kzr_build",
            "llext_binary_path": null,
            "uart_port": "/tmp/renode_uart_xofw4ov6/uart",
            "uart_baudrate": 115200,
            "uart_log_port": "/tmp/renode_uart_xofw4ov6/uart_log",
            "uart_log_baudrate": 115200,
            "auto_flash": false,
            "openocd_path": "openocd",
            "sensor": null,
            "number_of_batches": 16,
            "simulated": true,
            "runtime_binary_path": null,
            "platform_resc_path": "gh://antmicro:kenning-zephyr-runtime/renode/scripts/max32690evkit.resc;branch=main",
            "resc_dependencies": [],
            "post_start_commands": [],
            "disable_opcode_counters": false,
            "disable_profiler": false,
            "profiler_dump_path": "/tmp/renode_profiler_1kh4c4k5.dump",
            "profiler_interval_step": 10.0,
            "runtime_init_log_msg": "Inference server started",
            "runtime_init_timeout": 30,
            "name": "max32690evkit/max32690/m4",
            "platforms_definitions": [
                "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/.venv/lib/python3.11/site-packages/kenning/resources/platforms/platforms.yml"
            ]
        }
    },
    "protocol": {
        "type": "kenning.protocols.uart.UARTProtocol",
        "parameters": {
            "port": "/tmp/renode_uart_xofw4ov6/uart",
            "baudrate": 115200,
            "error_recovery": true,
            "timeout": 30
        }
    },
    "model_wrapper": {
        "type": "kenning.modelwrappers.anomaly_detection.vae.PyTorchAnomalyDetectionVAE",
        "parameters": {
            "encoder_neuron_list": [
                34,
                8,
                14,
                40,
                33
            ],
            "decoder_neuron_list": [
                44,
                15,
                36,
                25,
                16
            ],
            "latent_dim": 12,
            "hidden_activation": "softmax",
            "output_activation": "tanh",
            "batch_norm": true,
            "dropout_rate": 0.4054920288479,
            "loss_beta": 0.4235748410695,
            "loss_capacity": 0.7909717876419,
            "clip_grad_max_norm": 8.5833348167468,
            "batch_size": null,
            "learning_rate": null,
            "num_epochs": null,
            "evaluate": true,
            "model_path": "workspace/automl-results/1234_29_5.0.pth",
            "model_name": null
        }
    },
    "runtime": {
        "type": "kenning.runtimes.tflite.TFLiteRuntime",
        "parameters": {
            "save_model_path": "workspace/automl-results/vae.4.tflite",
            "delegates_list": null,
            "num_threads": 4,
            "llext_binary_path": null,
            "disable_performance_measurements": false
        }
    },
    "runtime_builder": {
        "type": "kenning.runtimebuilders.zephyr.ZephyrRuntimeBuilder",
        "parameters": {
            "board": "max32690evkit/max32690/m4",
            "application_dir": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/app",
            "build_dir": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/build",
            "venv_dir": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/.west-venv",
            "extra_targets": [
                "board-repl"
            ],
            "extra_build_args": [],
            "use_llext": false,
            "workspace": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime",
            "output_path": "workspace/kzr_build",
            "model_framework": "tflite"
        }
    }
}

```
## AutoML statistics


- *Optimized metric*: **f1**

- *The number of generated models*: **43**

- *The number of trained and evaluated models*: **26**

- *The number of successful training processes*: **33**

- *The number of models that caused a crash*: **0**

- *The number of models that failed due to the timeout*: **0**

- *The number of models that failed due to the too large size*: **10**



### Training overview

```{figure} generated/img/training_plot.*
---
name: automl_training_plot
alt: Loss value during AutoML training process
align: center
---

Loss value during AutoML training process
```

```{figure} generated/img/comparison_training_plot.*
---
name: automl_comparison_training_plot
alt: Comparison of loss value across models
align: center
---

Comparison of loss value across models
```



### Summary of generated models




```{figure} generated/img/trained_models_plot.*
---
name: automl_trained_models_plot
alt: Metrics of models trained by AutoML flow
align: center
---

Metrics of models trained by AutoML flow
```





```{table} Summary of generated models' parameters
---
align: center
---

| Model ID |  Number of layers | Optimized model size [KB] | Total parameters | Trainable parameters |
|---| ---: | ---: | ---: | ---: |
| 3 |  7 | 17.8515625 | 2815 | 2814 |
| 4 |  10 | 56.0625 | 11623 | 11622 |
| 5 |  17 | 36.93359375 | 7498 | 7497 |
| 6 |  21 | 37.2578125 | 7613 | 7612 |
| 7 |  27 | 64.43359375 | 14094 | 14093 |
| 8 |  14 | 57.20703125 | 11841 | 11840 |
| 9 |  21 | 40.7578125 | 7834 | 7833 |
| 10 |  17 | 41.953125 | 8732 | 8731 |
| 11 |  23 | 57.3125 | 11691 | 11690 |
| 12 |  21 | 37.65234375 | 7656 | 7655 |
| 13 |  17 | 37.7734375 | 7799 | 7798 |
| 14 |  12 | 41.64453125 | 8227 | 8226 |
| 15 |  10 | 45.86328125 | 10264 | 10263 |
| 16 |  13 | 48.98828125 | 10455 | 10454 |
| 17 |  8 | 22.78515625 | 4364 | 4363 |
| 18 |  27 | 50.55078125 | 10720 | 10719 |
| 19 |  11 | 63.03125 | 13723 | 13722 |
| 20 |  14 | 37.3203125 | 6909 | 6908 |
| 21 |  13 | 37.41796875 | 7850 | 7849 |
| 22 |  21 | 36.9609375 | 7176 | 7175 |
| 23 |  14 | 37.73828125 | 6924 | 6923 |
| 24 |  25 | 43.26953125 | 8756 | 8755 |
| 25 |  19 | 63.13671875 | 14181 | 14180 |
| 26 |  25 | 67.61328125 | 15361 | 15360 |
| 27 |  19 | 29.7734375 | 5708 | 5707 |
| 28 |  17 | 43.62890625 | 9106 | 9105 |
| 29 |  23 | 49.35546875 | 10270 | 10269 |
| 30 |  13 | 23.16015625 | 4384 | 4383 |
| 31 |  13 | 66.67578125 | 14396 | 14395 |
| 32 |  11 | 45.125 | 9425 | 9424 |
| 33 |  27 | 61.9609375 | 13772 | 13771 |
| 34 |  8 | 28.6171875 | 5457 | 5456 |
| 35 |  12 | 34.8125 | 6371 | 6370 |
| 36 |  13 | 68.93359375 | 14976 | 14975 |
| 37 |  11 | 45.89453125 | 9386 | 9385 |
| 38 |  10 | 47.0078125 | 10000 | 9999 |

```

## Classification comparison

### Comparison of inference time, F1 score and model size

```{figure} generated/img/accuracy_vs_inference_time.*
---
name: classification_performance_and_reporttypesrenode_of_workspaceautomlresults1234_3_50measurementsjson_classification_size_inference
alt: F1 score vs Inference time vs RAM usage
align: center
---

Model size, speed and quality summary.
The F1 score of the model is presented on Y axis.
The inference time of the model is presented on X axis.
The size of the model is represented by the size of its point.
```
```{list-table} Comparison of model inference time, accuracy and size
---
header-rows: 1
align: center
---

* - Model name
  - Mean Inference time [s]
  - Size [MB]
  - F1 score

* - workspace.automl-results.1234_3_5.0.measurements.json
  - 0.000560
  - 0.018
  - 0.250000

* - workspace.automl-results.1234_12_5.0.measurements.json
  - 0.001247
  - 0.040
  - 0.250000

* - workspace.automl-results.1234_21_5.0.measurements.json
  - 0.001094
  - 0.039
  - 0.133333

* - workspace.automl-results.1234_29_1.6666666666666665.measurements.json
  - 0.001449
  - 0.051
  - 0.250000

* - workspace.automl-results.1234_29_5.0.measurements.json
  - 0.001452
  - 0.051
  - 0.250000

```

### Detailed metrics comparison

```{figure} generated/img/classification_metric_comparison.*
---
name: classification_performance_and_reporttypesrenode_of_workspaceautomlresults1234_3_50measurementsjson_classification_metrics_radar
alt: Metric comparison
align: center
width: 100%
figclass: prevent-redirection
---

Radar chart representing the accuracy, precision and recall for models
```

```{list-table} Summary of classification metrics for models
---
header-rows: 1
align: center
---

* - Model name
  - Accuracy
  - Mean precision
  - Mean sensitivity
  - G-mean
  - ROC AUC
  - F1 score

* - workspace.automl-results.1234_3_5.0.measurements.json
  - **0.952000**
  - **0.729675**
  - **0.579132**
  - **0.406529**
  - **0.579132**
  - **0.250000**

* - workspace.automl-results.1234_12_5.0.measurements.json
  - **0.952000**
  - **0.729675**
  - **0.579132**
  - **0.406529**
  - **0.579132**
  - **0.250000**

* - workspace.automl-results.1234_21_5.0.measurements.json
  - 0.948000
  - 0.644399
  - 0.537465
  - 0.287460
  - 0.537465
  - 0.133333

* - workspace.automl-results.1234_29_1.6666666666666665.measurements.json
  - **0.952000**
  - **0.729675**
  - **0.579132**
  - **0.406529**
  - **0.579132**
  - **0.250000**

* - workspace.automl-results.1234_29_5.0.measurements.json
  - **0.952000**
  - **0.729675**
  - **0.579132**
  - **0.406529**
  - **0.579132**
  - **0.250000**

```

## Inference comparison

### Performance metrics



```{figure} generated/img/inference_step_comparison.*
---
name: classification_performance_and_reporttypesrenode_of_workspaceautomlresults1234_3_50measurementsjson_inference_step_comparison
alt: Inference time comparison
align: center
---

Plot represents changes of inference time over time for all models.
```

```{list-table} Summary of inference time metrics for models
---
header-rows: 1
align: center
---


* - Model name
  - Standard deviation [s]
  - Maximum [s]
  - Median [s]
  - Mean [s]
  - Minimum [s]
* - workspace.automl-results.1234_3_5.0.measurements.json
  - 0.000014
  - 0.000654
  - 0.000560
  - 0.000560
  - 0.000507
* - workspace.automl-results.1234_12_5.0.measurements.json
  - 0.000015
  - 0.001294
  - 0.001249
  - 0.001247
  - 0.001195
* - workspace.automl-results.1234_21_5.0.measurements.json
  - 0.000015
  - 0.001141
  - 0.001096
  - 0.001094
  - 0.001041
* - workspace.automl-results.1234_29_1.6666666666666665.measurements.json
  - 0.000015
  - 0.001511
  - 0.001452
  - 0.001449
  - 0.001396
* - workspace.automl-results.1234_29_5.0.measurements.json
  - 0.000015
  - 0.001496
  - 0.001452
  - 0.001452
  - 0.001396


```










### Mean comparison plots

```{figure} generated/img/mean_performance_comparison.*
---
name: classification_performance_and_reporttypesrenode_of_workspaceautomlresults1234_3_50measurementsjson_performance_comparison
alt: Performance comparison
align: center
---
Violin chart representing distribution of values for performance metrics for models
```

```{list-table} Performance metric for models
---
header-rows: 1
align: center
---
* - Model name
  - Inference time [s]
* - workspace.automl-results.1234_3_5.0.measurements.json
  - 0.000560
* - workspace.automl-results.1234_12_5.0.measurements.json
  - 0.001247
* - workspace.automl-results.1234_21_5.0.measurements.json
  - 0.001094
* - workspace.automl-results.1234_29_1.6666666666666665.measurements.json
  - 0.001449
* - workspace.automl-results.1234_29_5.0.measurements.json
  - 0.001452
```

## Renode performance measurements
### Executed instructions counters
```{figure} generated/img/executed_instructions_cpu0_plot_comparison.*
---
name: _cpu0_executedinstrplotpath_persecond
alt: Count of executed instructions per second for cpu0
align: center
---

Count of executed instructions per second for cpu0 during benchmark
```

```{figure} generated/img/cumulative_executed_instructions_cpu0_plot_comparison.*
---
name: _cpu0_executedinstrplotpath_cumulative
alt: Cumulative count of executed instructions for cpu0
align: center
---

Cumulative count of executed instructions for cpu0 during benchmark
```
### Memory access counters
```{figure} generated/img/memory_reads_plot_comparison.*
---
name: _memoryreadsplotpath_persecond
alt: Count of memory reads per second
align: center
---

Count of memory reads per second during benchmark
```

```{figure} generated/img/cumulative_memory_reads_plot_comparison.*
---
name: _memoryreadsplotpath_cumulative
alt: Cumulative count of memory reads
align: center
---

Cumulative count of memory reads during benchmark
```
```{figure} generated/img/memory_writes_plot_comparison.*
---
name: _memorywritessplotpath_persecond
alt: Count of memory writes per second
align: center
---

Count of memory writes per second during benchmark
```

```{figure} generated/img/cumulative_memory_writes_plot_comparison.*
---
name: _memorywritessplotpath_cumulative
alt: Cumulative count of memory writes
align: center
---

Cumulative count of memory writes during benchmark
```
### Peripheral access counters
```{figure} generated/img/nvic0_reads_plot_comparison.*
---
name: _nvic0_peripheralreadsplotpath_persecond
alt: Count of nvic0 reads per second
align: center
---

Count of nvic0 reads per second during benchmark
```

```{figure} generated/img/cumulative_nvic0_reads_plot_comparison.*
---
name: _nvic0_peripheralreadsplotpath_cumulative
alt: Cumulative count of nvic0 reads
align: center
---

Cumulative count of nvic0 reads during benchmark
```

```{figure} generated/img/nvic0_writes_plot_comparison.*
---
name: _nvic0_peripheralwritesplotpath_persecond
alt: Count of nvic0 writes per second
align: center
---

Count of nvic0 writes per second during benchmark
```

```{figure} generated/img/cumulative_nvic0_writes_plot_comparison.*
---
name: _nvic0_peripheralwritesplotpath_cumulative
alt: Cumulative count of nvic0 writes
align: center
---

Cumulative count of nvic0 writes during benchmark
```
```{figure} generated/img/uart2_reads_plot_comparison.*
---
name: _uart2_peripheralreadsplotpath_persecond
alt: Count of uart2 reads per second
align: center
---

Count of uart2 reads per second during benchmark
```

```{figure} generated/img/cumulative_uart2_reads_plot_comparison.*
---
name: _uart2_peripheralreadsplotpath_cumulative
alt: Cumulative count of uart2 reads
align: center
---

Cumulative count of uart2 reads during benchmark
```

```{figure} generated/img/uart2_writes_plot_comparison.*
---
name: _uart2_peripheralwritesplotpath_persecond
alt: Count of uart2 writes per second
align: center
---

Count of uart2 writes per second during benchmark
```

```{figure} generated/img/cumulative_uart2_writes_plot_comparison.*
---
name: _uart2_peripheralwritesplotpath_cumulative
alt: Cumulative count of uart2 writes
align: center
---

Cumulative count of uart2 writes during benchmark
```
```{figure} generated/img/uart0_reads_plot_comparison.*
---
name: _uart0_peripheralreadsplotpath_persecond
alt: Count of uart0 reads per second
align: center
---

Count of uart0 reads per second during benchmark
```

```{figure} generated/img/cumulative_uart0_reads_plot_comparison.*
---
name: _uart0_peripheralreadsplotpath_cumulative
alt: Cumulative count of uart0 reads
align: center
---

Cumulative count of uart0 reads during benchmark
```

```{figure} generated/img/uart0_writes_plot_comparison.*
---
name: _uart0_peripheralwritesplotpath_persecond
alt: Count of uart0 writes per second
align: center
---

Count of uart0 writes per second during benchmark
```

```{figure} generated/img/cumulative_uart0_writes_plot_comparison.*
---
name: _uart0_peripheralwritesplotpath_cumulative
alt: Cumulative count of uart0 writes
align: center
---

Cumulative count of uart0 writes during benchmark
```
### Exceptions counters

```{figure} generated/img/exceptions_plot_comparison.*
---
name: _exceptionsplotpath_persecond
alt: Count of raised exceptions per second
align: center
---

Count of raised exceptions per second during benchmark
```

```{figure} generated/img/cumulative_exceptions_plot_comparison.*
---
name: _exceptionsplotpath_cumulative
alt: Cumulative count of raised exceptions
align: center
---

Cumulative count of raised exceptions during benchmark
```
