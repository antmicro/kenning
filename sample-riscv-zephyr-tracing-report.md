
# Sample RISC-V Zephyr Tracing Report


### Commands used

````{note}

This section was generated using:

```bash
python -m kenning.__init__ \
    optimize \
    test \
    report \
    --cfg \
        /home/runner/work/kenning/kenning/scripts/configs/zephyr-tvm-magic-wand-hifive-unmatched-report.yml \
    --measurements \
        ./results.json \
    --report-path \
        /home/runner/work/kenning/kenning/docs/source/generated/sample-riscv-zephyr-tracing-report.md \
    --root-dir \
        /home/runner/work/kenning/kenning/docs/source/ \
    --img-dir \
        /home/runner/work/kenning/kenning/docs/source/generated/img/ \
    --report-name \
        Sample RISC-V Zephyr Tracing Report \
    --verbosity \
        INFO \
    --to-html


```
````


### General information for results.json

*Model framework*:

* torch ver. 2.8.0+cu128

*Input JSON*:

```json
{
    "dataset": {
        "type": "kenning.datasets.magic_wand_dataset.MagicWandDataset",
        "parameters": {
            "window_size": 128,
            "window_shift": 128,
            "noise_level": 20,
            "dataset_root": "build/data",
            "inference_batch_size": 1,
            "download_dataset": true,
            "force_download_dataset": false,
            "external_calibration_dataset": null,
            "split_fraction_test": 0.2,
            "split_fraction_val": null,
            "split_seed": 1234,
            "reduce_dataset": 1
        }
    },
    "dataconverter": {
        "type": "kenning.dataconverters.modelwrapper_dataconverter.ModelWrapperDataConverter",
        "parameters": {}
    },
    "optimizers": [
        {
            "type": "kenning.optimizers.tvm.TVMCompiler",
            "parameters": {
                "model_framework": "any",
                "target": null,
                "target_attrs": "",
                "target_microtvm_board": "hifive_unmatched/fu740/u74",
                "target_host": null,
                "zephyr_header_template": "gh://antmicro:kenning-zephyr-runtime/lib/kenning_inference_lib/runtimes/tvm/generated/model_impl.h.template;branch=main",
                "zephyr_llext_source_template": null,
                "opt_level": 3,
                "libdarknet_path": "/usr/local/lib/libdarknet.so",
                "compile_use_vm": false,
                "output_conversion_function": "default",
                "conv2d_data_layout": "",
                "conv2d_kernel_layout": "",
                "use_fp16_precision": false,
                "use_int8_precision": false,
                "int8_calibrate_chunk_by": -1,
                "use_tensorrt": false,
                "dataset_percentage": 0.25,
                "module_name": null,
                "compiled_model_path": "./build/compiled-model-magic-wand.graph_data",
                "location": "host"
            }
        }
    ],
    "platform": {
        "type": "kenning.platforms.zephyr.ZephyrPlatform",
        "parameters": {
            "zephyr_build_path": "build/hifive_unmatched",
            "llext_binary_path": null,
            "sensors": null,
            "sensors_frequency": null,
            "enable_zephelin_gdb": false,
            "enable_zephelin": true,
            "zephyr_base": null,
            "uart_port": "/tmp/renode_uart_cl1xv6y8/uart",
            "uart_baudrate": 115200,
            "uart_log_port": "/tmp/renode_uart_cl1xv6y8/uart_log",
            "uart_log_baudrate": 115200,
            "auto_flash": false,
            "openocd_path": "openocd",
            "sensor": null,
            "number_of_batches": 16,
            "simulated": true,
            "runtime_binary_path": null,
            "platform_resc_path": null,
            "resc_dependencies": [],
            "post_start_commands": [
                "logLevel 3 sysbus.uart1"
            ],
            "disable_opcode_counters": false,
            "disable_profiler": false,
            "profiler_dump_path": "/tmp/renode_profiler_qwu7i7tw.dump",
            "profiler_interval_step": 10.0,
            "runtime_init_log_msg": "Inference server started",
            "runtime_init_timeout": 30,
            "gdb_port": 3333,
            "renode_log_lines_single_read_limit": 5000,
            "name": "hifive_unmatched/fu740/u74",
            "platforms_definitions": [
                "kenning:///platforms/platforms.yml",
                "/home/runner/work/kenning/kenning/kenning/resources/platforms/platforms.yml"
            ]
        }
    },
    "protocol": {
        "type": "kenning.protocols.uart.UARTProtocol",
        "parameters": {
            "port": "/tmp/renode_uart_cl1xv6y8/uart",
            "baudrate": 115200,
            "error_recovery": true,
            "timeout": 30
        }
    },
    "model_wrapper": {
        "type": "kenning.modelwrappers.classification.pytorch_magic_wand.PyTorchMagicWandModelWrapper",
        "parameters": {
            "batch_size": null,
            "learning_rate": null,
            "num_epochs": null,
            "window_size": 128,
            "logdir": null,
            "export_dict": false,
            "model_path": "kenning:///models/classification/magic_wand.pth",
            "model_name": null
        }
    },
    "runtime": {
        "type": "kenning.runtimes.tvm.TVMRuntime",
        "parameters": {
            "save_model_path": "./build/compiled-model-magic-wand.graph_data",
            "target_device_context": "cpu",
            "target_device_context_id": 0,
            "runtime_use_vm": false,
            "llext_binary_path": null,
            "batch_size": 1,
            "disable_performance_measurements": false
        }
    },
    "runtime_builder": {
        "type": "kenning.runtimebuilders.zephyr.ZephyrRuntimeBuilder",
        "parameters": {
            "board": "hifive_unmatched/fu740/u74",
            "application_dir": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/app",
            "build_dir": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/build",
            "venv_dir": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime/.west-venv",
            "extra_targets": [
                "board-repl"
            ],
            "extra_build_args": [
                "-DCONFIG_KENNING_TVM_MODEL_PRE_GEN=y"
            ],
            "use_llext": false,
            "run_west_update": false,
            "workspace": "/home/runner/work/kenning/kenning/zephyr-workspace/kenning-zephyr-runtime",
            "output_path": "build/hifive_unmatched",
            "model_framework": "tvm"
        }
    }
}

```
## Inference quality metrics for results.json


```{figure} generated/img/confusion_matrix.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_confusionmatrix
alt: Confusion matrix
align: center
---

Confusion matrix
```

```{list-table} Inference quality metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Value
* - *Accuracy*
  - **0.992857**
* - *Mean precision*
  - **0.994681**
* - *Mean sensitivity*
  - **0.994681**
* - *G-mean*
  - **0.994638**
```
## Inference performance metrics for results.json


### Inference time

```{figure} generated/img/inference_time.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_inferencetime
alt: Inference time
align: center
---

Inference time
```

```{list-table} Inference time metrics
---
header-rows: 1
align: center
---

* - Statistic
  - Time [s]

* - First inference duration
  - **0.010179**
* - Mean
  - **0.009859**
* - Median
  - **0.009875**
* - Standard deviation
  - **0.000312**
* - Minimum
  - **0.009145**
* - Maximum
  - **0.010878**
```









## Renode performance measurements  for results.json


### Count of instructions used during inference

```{figure} generated/img/instr_barplot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_instrbarplot
alt: Count of used instructions during inference
align: center
---

Histogram of used instructions during inference
```
### Executed instructions counters
```{figure} generated/img/executed_instructions_cpu0_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_cpu0_executedinstrplotpath_persecond
alt: Count of executed instructions per second for cpu0
align: center
---

Count of executed instructions per second for cpu0 during benchmark
```

```{figure} generated/img/cumulative_executed_instructions_cpu0_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_cpu0_executedinstrplotpath_cumulative
alt: Cumulative count of executed instructions for cpu0
align: center
---

Cumulative count of executed instructions for cpu0 during benchmark
```
```{figure} generated/img/executed_instructions_cpu1_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_cpu1_executedinstrplotpath_persecond
alt: Count of executed instructions per second for cpu1
align: center
---

Count of executed instructions per second for cpu1 during benchmark
```

```{figure} generated/img/cumulative_executed_instructions_cpu1_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_cpu1_executedinstrplotpath_cumulative
alt: Cumulative count of executed instructions for cpu1
align: center
---

Cumulative count of executed instructions for cpu1 during benchmark
```
```{figure} generated/img/executed_instructions_cpu2_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_cpu2_executedinstrplotpath_persecond
alt: Count of executed instructions per second for cpu2
align: center
---

Count of executed instructions per second for cpu2 during benchmark
```

```{figure} generated/img/cumulative_executed_instructions_cpu2_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_cpu2_executedinstrplotpath_cumulative
alt: Cumulative count of executed instructions for cpu2
align: center
---

Cumulative count of executed instructions for cpu2 during benchmark
```
```{figure} generated/img/executed_instructions_cpu3_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_cpu3_executedinstrplotpath_persecond
alt: Count of executed instructions per second for cpu3
align: center
---

Count of executed instructions per second for cpu3 during benchmark
```

```{figure} generated/img/cumulative_executed_instructions_cpu3_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_cpu3_executedinstrplotpath_cumulative
alt: Cumulative count of executed instructions for cpu3
align: center
---

Cumulative count of executed instructions for cpu3 during benchmark
```
### Peripheral access counters
```{figure} generated/img/_clint_reads_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_clint_peripheralreadsplotpath_persecond
alt: Count of clint reads per second
align: center
---

Count of clint reads per second during benchmark
```

```{figure} generated/img/cumulative_clint_reads_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_clint_peripheralreadsplotpath_cumulative
alt: Cumulative count of clint reads
align: center
---

Cumulative count of clint reads during benchmark
```

```{figure} generated/img/_clint_writes_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_clint_peripheralwritesplotpath_persecond
alt: Count of clint writes per second
align: center
---

Count of clint writes per second during benchmark
```

```{figure} generated/img/cumulative_clint_writes_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_clint_peripheralwritesplotpath_cumulative
alt: Cumulative count of clint writes
align: center
---

Cumulative count of clint writes during benchmark
```
```{figure} generated/img/_uart0_reads_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_uart0_peripheralreadsplotpath_persecond
alt: Count of uart0 reads per second
align: center
---

Count of uart0 reads per second during benchmark
```

```{figure} generated/img/cumulative_uart0_reads_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_uart0_peripheralreadsplotpath_cumulative
alt: Cumulative count of uart0 reads
align: center
---

Cumulative count of uart0 reads during benchmark
```

```{figure} generated/img/_uart0_writes_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_uart0_peripheralwritesplotpath_persecond
alt: Count of uart0 writes per second
align: center
---

Count of uart0 writes per second during benchmark
```

```{figure} generated/img/cumulative_uart0_writes_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_uart0_peripheralwritesplotpath_cumulative
alt: Cumulative count of uart0 writes
align: center
---

Cumulative count of uart0 writes during benchmark
```
```{figure} generated/img/_uart1_reads_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_uart1_peripheralreadsplotpath_persecond
alt: Count of uart1 reads per second
align: center
---

Count of uart1 reads per second during benchmark
```

```{figure} generated/img/cumulative_uart1_reads_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_uart1_peripheralreadsplotpath_cumulative
alt: Cumulative count of uart1 reads
align: center
---

Cumulative count of uart1 reads during benchmark
```

```{figure} generated/img/_uart1_writes_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_uart1_peripheralwritesplotpath_persecond
alt: Count of uart1 writes per second
align: center
---

Count of uart1 writes per second during benchmark
```

```{figure} generated/img/cumulative_uart1_writes_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_uart1_peripheralwritesplotpath_cumulative
alt: Cumulative count of uart1 writes
align: center
---

Cumulative count of uart1 writes during benchmark
```
### Exceptions counters

```{figure} generated/img/exceptions_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_exceptionsplotpath_persecond
alt: Count of raised exceptions per second
align: center
---

Count of raised exceptions per second during benchmark
```

```{figure} generated/img/cumulative_exceptions_plot.*
---
name: classification_performance_renode_stats_and_reporttypeszephyr_traces_of_resultsjsonresults.json_exceptionsplotpath_cumulative
alt: Cumulative count of raised exceptions
align: center
---

Cumulative count of raised exceptions during benchmark
```
### Instructions stats
* *Instructions counters per inference pass*: **1390102**
* *Top 10 instructions and counters per inference pass*:
    - *c.addi*: **210880**
    - *flw*: **160863**
    - *bne*: **95295**
    - *fsw*: **76555**
    - *fadd.s*: **66052**
    - *fmul.s*: **62272**
    - *addi*: **61304**
    - *c.mv*: **54837**
    - *c.flw*: **36021**
    - *ld*: **34444**
### Memory allocation stats
* *Total allocated*: **3091632**
* *Total freed*: **3033712**
* *Peak allocated*: **72256**
* *Compiled model size*: **22486.0**

Host memory refers to memory of the CPU controlling the accelerator, while device memory is the memory of the accelerator.
### Zephyr traces

<a target="blank" href="zephyr_traces_report.html">(open in a new tab)</a>
<iframe allowfullscreen="true" id="embedded_zephyr_traces" name="Zephyr Traces Report" src="zephyr_traces_report.html" width="100%" height="1200px" frameborder="0"></iframe>

