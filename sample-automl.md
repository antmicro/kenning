## AutoML statistics


- *Optimized metric*: **f1**

- *The number of generated models*: **65**

- *The number of trained and evaluated models*: **47**

- *The number of successful training processes*: **58**

- *The number of models that caused a crash*: **0**

- *The number of models that failed due to the timeout*: **0**

- *The number of models that failed due to the too large size*: **7**

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
| 3 |  7 | 15.78515625 | 2815 | 2814 |
| 4 |  10 | 49.8828125 | 11623 | 11622 |
| 5 |  17 | 33.10546875 | 7498 | 7497 |
| 6 |  21 | 33.703125 | 7613 | 7612 |
| 7 |  27 | 61.05078125 | 14094 | 14093 |
| 8 |  14 | 50.42578125 | 11841 | 11840 |
| 9 |  21 | 35.796875 | 7834 | 7833 |
| 10 |  17 | 37.453125 | 8732 | 8731 |
| 11 |  23 | 54.33984375 | 11691 | 11690 |
| 12 |  21 | 30.58984375 | 7656 | 7655 |
| 13 |  17 | 33.16015625 | 7799 | 7798 |
| 14 |  12 | 37.78125 | 8227 | 8226 |
| 15 |  10 | 42.41796875 | 10264 | 10263 |
| 16 |  13 | 45.55859375 | 10455 | 10454 |
| 17 |  8 | 19.86328125 | 4364 | 4363 |
| 18 |  27 | 47.53125 | 10720 | 10719 |
| 19 |  21 | 35.453125 | 8204 | 8203 |
| 20 |  9 | 17.77734375 | 3190 | 3189 |
| 21 |  13 | 32.6171875 | 7850 | 7849 |
| 22 |  12 | 29.12109375 | 5864 | 5863 |
| 23 |  15 | 34.7578125 | 8009 | 8008 |
| 24 |  25 | 39.62890625 | 8756 | 8755 |
| 25 |  7 | 40.13671875 | 9288 | 9287 |
| 26 |  11 | 34.09765625 | 6843 | 6842 |
| 27 |  12 | 57.0625 | 14867 | 14866 |
| 28 |  13 | 40.23046875 | 7844 | 7843 |
| 29 |  21 | 60.14453125 | 14766 | 14765 |
| 30 |  19 | 39.8125 | 9778 | 9777 |
| 31 |  19 | 53.26953125 | 12119 | 12118 |
| 32 |  10 | 27.86328125 | 5434 | 5433 |
| 33 |  12 | 36.3515625 | 8413 | 8412 |
| 34 |  17 | 51.25390625 | 12285 | 12284 |
| 35 |  17 | 47.73828125 | 11113 | 11112 |
| 36 |  23 | 31.2109375 | 6730 | 6729 |
| 37 |  11 | 29.9453125 | 6124 | 6123 |
| 38 |  25 | 38.5625 | 8307 | 8306 |
| 39 |  13 | 61.4765625 | 14976 | 14975 |
| 40 |  21 | 38.625 | 9280 | 9279 |
| 41 |  10 | 38.50390625 | 10000 | 9999 |
| 42 |  9 | 30.87890625 | 7144 | 7143 |
| 43 |  17 | 46.85546875 | 11825 | 11824 |
| 44 |  19 | 16.26953125 | 2494 | 2493 |
| 45 |  15 | 16.72265625 | 3006 | 3005 |
| 46 |  19 | 42.20703125 | 9575 | 9574 |
| 47 |  15 | 22.12109375 | 4414 | 4413 |
| 48 |  23 | 31.94921875 | 6487 | 6486 |
| 49 |  15 | 46.23046875 | 10910 | 10909 |
| 50 |  12 | 40.47265625 | 8755 | 8754 |
| 51 |  15 | 37.50390625 | 10665 | 10664 |
| 52 |  17 | 44.21875 | 10603 | 10602 |
| 53 |  9 | 37.53515625 | 9222 | 9221 |
| 54 |  13 | 37.64453125 | 8871 | 8870 |
| 55 |  25 | 31.11328125 | 6490 | 6489 |
| 56 |  19 | 53.01171875 | 13504 | 13503 |

```

## Classification comparison

### Comparison of inference time, F1 score and model size

```{figure} generated/img/f1_vs_inference_time.*
---
name: classification_and_performance_of_automl_conf_0_classification_size_inference
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

* - automl_conf_0
  - 0.001215
  - 0.045
  - 0.470588

* - automl_conf_1
  - 0.001049
  - 0.041
  - 0.533333

* - automl_conf_2
  - 0.001048
  - 0.041
  - 0.533333

* - automl_conf_3
  - 0.001020
  - 0.035
  - 0.533333

* - automl_conf_4
  - 0.000938
  - 0.037
  - 0.533333

```

### Detailed metrics comparison

```{figure} generated/img/classification_metric_comparison.*
---
name: classification_and_performance_of_automl_conf_0_classification_metrics_radar
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

* - automl_conf_0
  - 0.964000
  - 0.713923
  - 0.739669
  - 0.699764
  - 0.739669
  - 0.470588

* - automl_conf_1
  - **0.972000**
  - **0.777484**
  - **0.743802**
  - **0.702710**
  - **0.743802**
  - **0.533333**

* - automl_conf_2
  - **0.972000**
  - **0.777484**
  - **0.743802**
  - **0.702710**
  - **0.743802**
  - **0.533333**

* - automl_conf_3
  - **0.972000**
  - **0.777484**
  - **0.743802**
  - **0.702710**
  - **0.743802**
  - **0.533333**

* - automl_conf_4
  - **0.972000**
  - **0.777484**
  - **0.743802**
  - **0.702710**
  - **0.743802**
  - **0.533333**

```

## Inference comparison

### Performance metrics



```{figure} generated/img/inference_step_comparison.*
---
name: classification_and_performance_of_automl_conf_0_inference_step_comparison
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
  - Standard deviation [s]
  - Maximum [s]
  - Mean [s]
* - automl_conf_0
  - 0.001212
  - 0.001187
  - 0.000016
  - 0.001333
  - 0.001215
* - automl_conf_1
  - 0.001046
  - 0.001028
  - 0.000010
  - 0.001100
  - 0.001049
* - automl_conf_2
  - 0.001046
  - 0.001022
  - 0.000008
  - 0.001105
  - 0.001048
* - automl_conf_3
  - 0.001017
  - 0.000991
  - 0.000009
  - 0.001067
  - 0.001020
* - automl_conf_4
  - 0.000936
  - 0.000928
  - 0.000008
  - 0.001001
  - 0.000938


```










### Mean comparison plots

```{figure} generated/img/mean_performance_comparison.*
---
name: classification_and_performance_of_automl_conf_0_performance_comparison
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
* - automl_conf_0
  - 0.001215
* - automl_conf_1
  - 0.001049
* - automl_conf_2
  - 0.001048
* - automl_conf_3
  - 0.001020
* - automl_conf_4
  - 0.000938
```

