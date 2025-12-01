## AutoML statistics


- *Optimized metric*: **f1**

- *The number of generated models*: **40**

- *The number of trained and evaluated models*: **24**

- *The number of successful training processes*: **31**

- *The number of models that caused a crash*: **0**

- *The number of models that failed due to the timeout*: **0**

- *The number of models that failed due to the too large size*: **9**

- *The number of models that failed due to incompatibility*: **0**



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
| 19 |  13 | 22.328125 | 4046 | 4045 |
| 20 |  19 | 49.92578125 | 10707 | 10706 |
| 21 |  19 | 44.93359375 | 9318 | 9317 |
| 22 |  19 | 38.32421875 | 7967 | 7966 |
| 23 |  21 | 24.71484375 | 4251 | 4250 |
| 24 |  19 | 57.14453125 | 12877 | 12876 |
| 25 |  25 | 74.7265625 | 16820 | 16819 |
| 26 |  23 | 49.66015625 | 10292 | 10291 |
| 27 |  23 | 37.51953125 | 7311 | 7310 |
| 28 |  23 | 38.56640625 | 7591 | 7590 |
| 29 |  13 | 30.2109375 | 5802 | 5801 |
| 30 |  17 | 24.7109375 | 4206 | 4205 |
| 31 |  9 | 27.79296875 | 5493 | 5492 |
| 32 |  15 | 69.015625 | 14880 | 14879 |
| 33 |  11 | 30.69140625 | 6079 | 6078 |
| 34 |  14 | 64.64453125 | 13745 | 13744 |
| 35 |  8 | 59.04296875 | 13616 | 13615 |

```

## Classification comparison

### Comparison of inference time, F1 score and model size

```{figure} generated/img/accuracy_vs_inference_time.*
---
name: classification_performance_and_reporttypesrenode_of_workspaceautomlresults1234_27_50measurementsjson_classification_size_inference
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

* - workspace.automl-results.1234_27_5.0.measurements.json
  - 0.001214
  - 0.038
  - 0.375000

* - workspace.automl-results.1234_30_1.6666666666666665.measurements.json
  - 0.000962
  - 0.026
  - 0.333333

* - workspace.automl-results.1234_30_5.0.measurements.json
  - 0.000959
  - 0.026
  - 0.333333

* - workspace.automl-results.1234_3_5.0.measurements.json
  - 0.000560
  - 0.018
  - 0.250000

* - workspace.automl-results.1234_12_5.0.measurements.json
  - 0.001248
  - 0.040
  - 0.250000

```

### Detailed metrics comparison

```{figure} generated/img/classification_metric_comparison.*
---
name: classification_performance_and_reporttypesrenode_of_workspaceautomlresults1234_27_50measurementsjson_classification_metrics_radar
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

* - workspace.automl-results.1234_27_5.0.measurements.json
  - **0.960000**
  - **0.856707**
  - **0.622899**
  - **0.498948**
  - **0.622899**
  - **0.375000**

* - workspace.automl-results.1234_30_1.6666666666666665.measurements.json
  - 0.952000
  - 0.731557
  - 0.618697
  - 0.496839
  - 0.618697
  - 0.333333

* - workspace.automl-results.1234_30_5.0.measurements.json
  - 0.952000
  - 0.731557
  - 0.618697
  - 0.496839
  - 0.618697
  - 0.333333

* - workspace.automl-results.1234_3_5.0.measurements.json
  - 0.952000
  - 0.729675
  - 0.579132
  - 0.406529
  - 0.579132
  - 0.250000

* - workspace.automl-results.1234_12_5.0.measurements.json
  - 0.952000
  - 0.729675
  - 0.579132
  - 0.406529
  - 0.579132
  - 0.250000

```

## Inference comparison

### Performance metrics



```{figure} generated/img/inference_step_comparison.*
---
name: classification_performance_and_reporttypesrenode_of_workspaceautomlresults1234_27_50measurementsjson_inference_step_comparison
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
  - Median [s]
  - Minimum [s]
  - Mean [s]
  - Maximum [s]
  - Standard deviation [s]
* - workspace.automl-results.1234_27_5.0.measurements.json
  - 0.001204
  - 0.001155
  - 0.001214
  - 0.001295
  - 0.000030
* - workspace.automl-results.1234_30_1.6666666666666665.measurements.json
  - 0.000951
  - 0.000920
  - 0.000962
  - 0.001057
  - 0.000027
* - workspace.automl-results.1234_30_5.0.measurements.json
  - 0.000951
  - 0.000916
  - 0.000959
  - 0.001025
  - 0.000025
* - workspace.automl-results.1234_3_5.0.measurements.json
  - 0.000550
  - 0.000518
  - 0.000560
  - 0.000652
  - 0.000026
* - workspace.automl-results.1234_12_5.0.measurements.json
  - 0.001241
  - 0.001203
  - 0.001248
  - 0.001333
  - 0.000025


```










### Mean comparison plots

```{figure} generated/img/mean_performance_comparison.*
---
name: classification_performance_and_reporttypesrenode_of_workspaceautomlresults1234_27_50measurementsjson_performance_comparison
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
* - workspace.automl-results.1234_27_5.0.measurements.json
  - 0.001214
* - workspace.automl-results.1234_30_1.6666666666666665.measurements.json
  - 0.000962
* - workspace.automl-results.1234_30_5.0.measurements.json
  - 0.000959
* - workspace.automl-results.1234_3_5.0.measurements.json
  - 0.000560
* - workspace.automl-results.1234_12_5.0.measurements.json
  - 0.001248
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
