# Structured Pruning of TensorFlow Models

This section contains scenarios with descriptions for pruning [`TensorFlowPetDatasetMobileNetV2`](https://github.com/antmicro/kenning/blob/main/kenning/modelwrappers/classification/tensorflow_pet_dataset.py).

TensorFlow pruning works during model's training and is executed every predefined number of steps.
It doesn't remove weights or change size of layers, so number of parameters or models size do not change (difference is visible after compressing model into e.g. ZIP archive).
Instead, pruned weights are set to zero, which can speed up inference on designated hardware and there is no restriction to what can be pruned - like to reduce layer's size only whole columns/rows of weight can be removed.

## Setup

A few additional dependencies are required to prune TensorFlow models, they can be installed using the following command:

```bash
pip install "kenning[tensorflow,reports] @ git+https://github.com/antmicro/kenning.git"
```

## Experiments

At the beginning, we would like to know a performance of the original model, which can be achieved by running the following pipeline.

```{literalinclude} ../scripts/jsonconfigs/mobilenetv2-tensorflow.json
:language: json
```

To test it, run:

```bash
kenning test --json-cfg scripts/jsonconfigs/mobilenetv2-tensorflow.json --measurements build/tf.json
```

`TensorflowPruningOptimizer` has two main parameters for adjusting pruning process:
* `target_sparsity` - defines sparsity of weights after the pruning,
* `prune_dense` - if `true` only dense layers will be pruned, otherwise whole model will be used.

There is also possibility to adjust how often model will be pruned (with `pruning_frequency`), and when it cannot be pruned (`pruning_end`).

Moreover, in this example we decided to train model for three epochs with size of the batch equals 128.
Apart from that, there is also possibility to chose `optimizer` (one of `adam`, `RMSprop` or `SGD`) and specify if network's output is normalized with `disable_from_logits`.

```{literalinclude} ../scripts/jsonconfigs/pruning-mobilenetv2-tensorflow.json
:language: json
:emphasize-lines: 15-26
```

To prune model, run:

```bash
kenning optimize --json-cfg scripts/jsonconfigs/pruning-mobilenetv2-tensorflow.json
kenning test --json-cfg scripts/jsonconfigs/pruning-mobilenetv2-tensorflow.json --measurements build/tf-pruning.json
```

Despite the fact, that Kenning CLI is capable of running commands in sequence (like `kenning optimize test [FLAGS]`), we suggest separating them to make sure performance measurements are more precise.

In order to compare model before and after pruning, report can be generated with the following command:

```bash
kenning report \
  --measurements \
    build/tf.json \
    build/tf-pruning.json \
  --report-path build/tf-pruning.md \
  --to-html
```

For greater size reduction, we can use larger sparsity with adjusted parameters, like:

```json
{
  "type": "kenning.optimizers.tensorflow_pruning.TensorFlowPruningOptimizer",
  "parameters": {
    "compiled_model_path": "./build/tf-pruning-0_3.h5",
    "target_sparsity": 0.3,
    "batch_size": 128,
    "epochs": 5,
    "pruning_frequency": 40,
    "pruning_end": 82,
    "save_to_zip": true
  }
}
```

Moreover, default TensorFlow version of pretrained MobileNetV2 (`kenning:///models/classification/tensorflow_pet_dataset_mobilenetv2.h5`) is exported with optimizer, what makes it a lot larger.
To ensure better quality of measurements, we will present data with default MobileNetV2 stripped from any unnecessary information.
Summary of pruned models:

| Sparsity      | Accuracy     | Compressed size | Size reduction |
|---------------|--------------|-----------------|----------------|
| ---           | 0.9605442177 |      15,508,373 |          0.00% |
| 0.15          | 0.9190476190 |      14,062,345 |          9.32% |
| 0.3           | 0.8367346939 |      12,349,076 |         20.37% |

```{figure} ../img/pruning-tf-classification-comparison.*
---
name: pruning-tf-classification-comparison
alt: Pruning classification comparison
align: center
---

Model size, speed and quality comparison for TensorFlow pruning
```

## (Optional) Clustering

{{project}} provides a few other methods for reducing size of the model.
For instance, clustering which groups similar weights in each layer and makes them equal.
It can be used by adding optimizer:

```{literalinclude} ../scripts/jsonconfigs/pruning-clustering-mobilenetv2-tensorflow.json
:language: json
:lines: 14-34
:emphasize-lines: 12-20
```

To run it, use:

```bash
kenning optimize test report --json-cfg scripts/jsonconfigs/pruning-clustering-mobilenetv2-tensorflow.json --measurements build/tf-all.json --report-path build/tf-pruning-clustering.md --to-html
```

Clustering allows to greatly reduce size of the model without reducing performance or resources usage.
Here is a comparison of model with and without clustering:

| Sparsity  | Number of clusters | Accuracy     | Compressed size | Size reduction |
|-----------|--------------------|--------------|-----------------|----------------|
| ---       | ---                | 0.9605442177 |      15,508,373 |          0.00% |
| 0.15      | ---                | 0.9190476190 |      14,062,345 |          9.32% |
| 0.15      | 60                 | 0.9006802721 |       4,451,142 |         71.30% |

```{figure} ../img/pruning-clustering-tf-classification-comparison.*
---
name: pruning-clustering-tf-classification-comparison
alt: Pruning classification comparison
align: center
---

Model size, speed and quality comparison for TensorFlow pruning and clustering
```

