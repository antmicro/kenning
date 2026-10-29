
## Pet Dataset classification using TVM-compiled TensorFlow model




### General information for build.local-cpu-tvm-tensorflow-classification.json

*Model framework*:

* keras ver. 2.21.0


````{dropdown} Used commands

This section was generated using:

```bash
python -m kenning.__init__ \
    optimize \
    test \
    --modelwrapper-cls \
        kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    --dataset-cls \
        kenning.datasets.pet_dataset.PetDataset \
    --measurements \
        ./build/local-cpu-tvm-tensorflow-classification.json \
    --compiler-cls \
        kenning.optimizers.tvm.TVMCompiler \
    --runtime-cls \
        kenning.runtimes.tvm.TVMRuntime \
    --model-path \
        kenning:///models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework \
        keras \
    --target \
        llvm \
    --compiled-model-path \
        ./build/compiled-model.tar \
    --opt-level \
        3 \
    --save-model-path \
        ./build/compiled-model.tar \
    --target-device-context \
        cpu \
    --dataset-root \
        ./build/PetDataset/ \
    --inference-batch-size \
        1 \
    --verbosity \
        INFO

python -m kenning.__init__ \
    report \
    --report-path \
        docs/source/generated/local-cpu-tvm-tensorflow-classification.md \
    --report-name \
        Pet Dataset classification using TVM-compiled TensorFlow model \
    --root-dir \
        docs/source/ \
    --img-dir \
        docs/source/generated/img \
    --report-types \
        performance \
        classification \
    --measurements \
        build/local-cpu-tvm-tensorflow-classification.json \
    --smaller-header

```
````
````{dropdown} Input scenario

```json
{
    "dataset": {
        "type": "kenning.datasets.pet_dataset.PetDataset",
        "parameters": {
            "classify_by": "breeds",
            "image_memory_layout": "NHWC",
            "dataset_root": "build/PetDataset",
            "inference_batch_size": 1,
            "download_dataset": true,
            "force_download_dataset": false,
            "external_calibration_dataset": null,
            "split_fraction_test": 0.2,
            "split_fraction_val": null,
            "split_seed": 1234,
            "reduce_dataset": 1.0
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
                "model_framework": "keras",
                "target": "llvm",
                "target_attrs": "",
                "target_microtvm_board": null,
                "target_host": null,
                "zephyr_header_template": null,
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
                "compiled_model_path": "./build/compiled-model.tar",
                "location": "host"
            }
        }
    ],
    "platform": {
        "type": "kenning.platforms.local.LocalPlatform",
        "parameters": {
            "name": null,
            "platforms_definitions": [
                "kenning:///platforms/platforms.yml",
                "/home/runner/work/kenning/kenning/kenning/resources/platforms/platforms.yml"
            ]
        }
    },
    "model_wrapper": {
        "type": "kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2",
        "parameters": {
            "batch_size": null,
            "learning_rate": null,
            "num_epochs": null,
            "logdir": null,
            "model_path": "kenning:///models/classification/tensorflow_pet_dataset_mobilenetv2.h5",
            "model_name": null
        }
    },
    "runtime": {
        "type": "kenning.runtimes.tvm.TVMRuntime",
        "parameters": {
            "save_model_path": "./build/compiled-model.tar",
            "target_device_context": "cpu",
            "target_device_context_id": 0,
            "runtime_use_vm": false,
            "llext_binary_path": null,
            "batch_size": 1,
            "disable_performance_measurements": false
        }
    }
}

```
````
## Inference performance metrics for build.local-cpu-tvm-tensorflow-classification.json


### Inference time

```{figure} generated/img/inference_time.*
---
name: performance_and_classification_of_buildlocalcputvmtensorflowclassificationjsonbuild.local-cpu-tvm-tensorflow-classification.json_inferencetime
alt: Inference time
align: center
---

Inference time
```

```{figure} generated/img/_pre_post_process.*
---
name: performance_and_classification_of_buildlocalcputvmtensorflowclassificationjsonbuild.local-cpu-tvm-tensorflow-classification.json_pre_post_process
alt: Preprocessing and postprocessing time
align: center
---

Preprocessing and postprocessing time
```


```{list-table} Inference time metrics
---
header-rows: 1
align: center
---

* - Statistic
  - Time [s]

* - First inference duration
  - **0.049278**
* - Mean
  - **0.047709**
* - Median
  - **0.047690**
* - Standard deviation
  - **0.000973**
* - Minimum
  - **0.041483**
* - Maximum
  - **0.055237**
```




### Average CPU usage by process

```{figure} generated/img/cpu_usage.*
---
name: performance_and_classification_of_buildlocalcputvmtensorflowclassificationjsonbuild.local-cpu-tvm-tensorflow-classification.json_cpuusage
alt: Average CPU usage by process
align: center
---

Average CPU usage by process during benchmark
```






### RAM usage by process

```{figure} generated/img/cpu_mem_usage.*
---
name: performance_and_classification_of_buildlocalcputvmtensorflowclassificationjsonbuild.local-cpu-tvm-tensorflow-classification.json_process_memusage
alt: RAM usage by process
align: center
---

RAM usage by process during benchmark
```

```{list-table} RAM usage by process metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Usage [%]

* - Mean
  - **8.130**
* - Standard deviation
  - **0.050**
* - Median
  - **8.158**

```














## Inference quality metrics for build.local-cpu-tvm-tensorflow-classification.json


```{figure} generated/img/confusion_matrix.*
---
name: performance_and_classification_of_buildlocalcputvmtensorflowclassificationjsonbuild.local-cpu-tvm-tensorflow-classification.json_confusionmatrix
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
  - **0.957143**
* - *Mean precision*
  - **0.958628**
* - *Mean sensitivity*
  - **0.956698**
* - *G-mean*
  - **0.955802**
* - *Top-5 accuracy*
  - **0.995918**
```
