# Structured pruning for PyTorch models

One of the methods of reducing model's size is pruning, which decreases number of neurons in layers by removing the least meaningful ones.
In this example, we present scenarios for pruning [`PyTorchPetDatasetMobileNetV2`](https://github.com/antmicro/kenning/blob/main/kenning/modelwrappers/classification/pytorch_pet_dataset.py) using [Neural Network Intelligence](https://github.com/microsoft/nni).

## Setup

Installation of required dependencies:

```bash
pip install "kenning[nni,reports] @ git+https://github.com/antmicro/kenning.git"
```

## Experiments

In order to compare original model, we have to execute the scenario:

```{literalinclude} ../scripts/jsonconfigs/mobilenetv2-pytorch.json
:language: json
```

To run it, use this command:

```bash
kenning test \
  --json-cfg scripts/jsonconfigs/mobilenetv2-pytorch.json \
  --measurements build/torch.json
```

{{project}} supports activation-based pruners, specific one can be chosen with `pruner_type` parameter:
* `apoz` - [`ActivationAPoZRankPruner`](https://nni.readthedocs.io/en/v2.5/Compression/v2_pruning_algo.html#activation-apoz-rank-pruner) based on Average Percentage of Zeros in activations,
* `mean_rank` - [`ActivationMeanRankPruner`](https://nni.readthedocs.io/en/v2.5/Compression/v2_pruning_algo.html#activation-mean-rank-pruner) based on metric that calculates the smallest mean value of activations.

These activations are collected during dataset inference and its number can be influenced by `training_steps`.
Moreover, pruning have two modes, `dependency_aware` which makes pruner aware of channels' and groups' dependencies.
On the other hand, in `normal` model, this information is ignored.
Also, there is a possibility to chose which `activation` pruner will use - `relu`, `relu6` or `gelu`.
Additional configuration can be specified in `config_list`, which follows the format defined in [NNI specification](https://nni.readthedocs.io/en/stable/compression/config_list.html#pruning-specific-configuration-keys).
Furthermore, if `exclude_last_layer` is positive, {{project}} will try to append configuration excluding last layer to the `config_list`, in order to prevent changing size of the output.
Apart from that, `confidence` defines coefficient for the sparsity inference and also the batch size of the dummy input for this process.
If GPU is available, it will be used by default, but as pruning can be memory-consuming, there is a `pruning_on_cuda` option for restricting GPU usage during this process.

Other arguments influence fine-tuning of the pruned model, e.g. `criterion` and `optimizer` accepts paths to the class respectively calculating criterion and optimizing neural network.
Also, number of `finetuning_epochs` can be changed, as well as `finetuning_batch_size` and `finetuning_learning_rate`.

```{literalinclude} ../scripts/jsonconfigs/pruning-mobilenetv2-pytorch.json
:language: json
:emphasize-lines: 16-44
```

Run pruning with:

```bash
kenning optimize test \
  --json-cfg scripts/jsonconfigs/pruning-mobilenetv2-pytorch.json \
  --measurements build/nni-pruning.json
```

To ensure better quality of performance measurements, we suggest running optimization and tests separately, like:

```bash test-skip
kenning optimize --json-cfg scripts/jsonconfigs/pruning-mobilenetv2-pytorch.json
kenning test \
  --json-cfg scripts/jsonconfigs/pruning-mobilenetv2-pytorch.json \
  --measurements build/nni-pruning.json
```

For greater size reduction, we can use larger sparsity with adjusted parameters, like:

{ emphasize-lines="7,14,18-19,24" }
```json
{
  "type": "kenning.optimizers.nni_pruning.NNIPruningOptimizer",
  "parameters": {
    "pruner_type": "mean_rank",
    "config_list": [
      {
        "total_sparsity": 0.15,
        "op_types": [
          "Conv2d",
          "Linear"
        ]
      }
    ],
    "training_steps": 92,
    "activation": "relu",
    "compiled_model_path": "build/nni-pruning-0_15.pth",
    "mode": "dependency_aware",
    "finetuning_epochs": 10,
    "finetuning_batch_size": 64,
    "confidence": 8,
    "criterion": "torch.nn.CrossEntropyLoss",
    "optimizer": "torch.optim.Adam",
    "pruning_on_cuda": true,
    "finetuning_learning_rate": 1e-04,
    "exclude_last_layer": true
  }
}
```

## Results

Models can be compared with generated report:

```bash
kenning report \
  --measurements \
    build/torch.json \
    build/nni-pruning.json \
  --report-path build/nni-pruning.md \
  --to-html
```

Pruned models have significantly fewer parameters, which results in decreased GPU and VRAM usage without increasing inference time.
Summary of a few examples with different sparsity:

| Sparsity      | Accuracy     | Number of parameters | Fine-tuning epochs | Size reduction |
|---------------|--------------|----------------------|--------------------|----------------|
| ---           | 0.9632653061 |            4,130,853 |                --- |          0.00% |
| 0.05          | 0.9299319728 |            3,660,421 |                  3 |         11.27% |
| 0.15          | 0.8102040816 |            2,813,380 |                 10 |         31.61% |

```{figure} ../img/pruning-nni-classification-comparison.*
---
name: pruning-nni-classification-comparison
alt: Pruning classification comparison
align: center
---

Model size, speed and quality comparison for NNI pruning
```

```{figure} ../img/pruning-nni-gpu-usage-comparison.*
---
name: pruning-nni-gpu-usage
alt: Pruning GPU usage comparison
align: center
---

Plot represents changes of GPU usage over time
```

```{figure} ../img/pruning-nni-gpu-mem-comparison.*
---
name: pruning-nni-gpu-mem
alt: Pruning GPU memory usage comparison
align: center
---

Plot represents changes of GPU RAM usage over time
```

