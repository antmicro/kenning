## AutoML statistics


- *Optimized metric*: **f1**

- *The number of generated models*: **45**

- *The number of trained and evaluated models*: **31**

- *The number of successful training processes*: **38**

- *The number of models that caused a crash*: **0**

- *The number of models that failed due to the timeout*: **1**

- *The number of models that failed due to the too large size*: **6**

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
| 3 |  7 | 15.9296875 | 2815 | 2814 |
| 4 |  10 | 51.984375 | 11623 | 11622 |
| 5 |  17 | 33.03125 | 7498 | 7497 |
| 6 |  21 | 33.50390625 | 7613 | 7612 |
| 7 |  27 | 60.47265625 | 14094 | 14093 |
| 8 |  14 | 50.34765625 | 11841 | 11840 |
| 9 |  21 | 38.08203125 | 7834 | 7833 |
| 10 |  17 | 37.18359375 | 8732 | 8731 |
| 11 |  23 | 56.640625 | 11691 | 11690 |
| 12 |  21 | 30.38671875 | 7656 | 7655 |
| 13 |  17 | 33.03125 | 7799 | 7798 |
| 14 |  12 | 37.92578125 | 8227 | 8226 |
| 15 |  10 | 42.484375 | 10264 | 10263 |
| 16 |  13 | 46.84375 | 10455 | 10454 |
| 17 |  8 | 19.9296875 | 4364 | 4363 |
| 18 |  27 | 47.04296875 | 10720 | 10719 |
| 19 |  21 | 35.25 | 8204 | 8203 |
| 20 |  9 | 17.921875 | 3190 | 3189 |
| 21 |  13 | 32.6796875 | 7850 | 7849 |
| 22 |  12 | 29.265625 | 5864 | 5863 |
| 23 |  15 | 34.55078125 | 8009 | 8008 |
| 24 |  25 | 39.16015625 | 8756 | 8755 |
| 25 |  7 | 41.44140625 | 9288 | 9287 |
| 26 |  11 | 36.46484375 | 6843 | 6842 |
| 27 |  12 | 57.12890625 | 14867 | 14866 |
| 28 |  13 | 43.13671875 | 7844 | 7843 |
| 29 |  21 | 62.4296875 | 14766 | 14765 |
| 30 |  19 | 39.640625 | 9778 | 9777 |
| 31 |  19 | 55.15234375 | 12119 | 12118 |
| 32 |  10 | 29.96484375 | 5434 | 5433 |
| 33 |  12 | 36.23828125 | 8413 | 8412 |
| 34 |  17 | 51.11328125 | 12285 | 12284 |
| 35 |  17 | 47.55859375 | 11113 | 11112 |
| 36 |  23 | 30.75 | 6730 | 6729 |
| 37 |  11 | 29.9296875 | 6124 | 6123 |
| 38 |  25 | 37.8203125 | 8307 | 8306 |
| 39 |  13 | 61.421875 | 14976 | 14975 |

```

## Classification comparison

### Comparison of inference time, F1 score and model size

```{figure} generated/img/f1_vs_inference_time.*
---
name: classification_and_reporttypesperformance_of_automl_conf_0_classification_size_inference
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
  - 0.001075
  - 0.040
  - 0.533333

* - automl_conf_1
  - 0.001074
  - 0.041
  - 0.533333

* - automl_conf_3
  - 0.000946
  - 0.037
  - 0.533333

* - automl_conf_4
  - 0.000407
  - 0.016
  - 0.533333

* - automl_conf_5
  - 0.000407
  - 0.016
  - 0.533333

```

### Detailed metrics comparison

```{figure} generated/img/classification_metric_comparison.*
---
name: classification_and_reporttypesperformance_of_automl_conf_0_classification_metrics_radar
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
  - **0.972000**
  - **0.777484**
  - **0.743802**
  - **0.702710**
  - **0.743802**
  - **0.533333**

* - automl_conf_1
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

* - automl_conf_5
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
name: classification_and_reporttypesperformance_of_automl_conf_0_inference_step_comparison
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
  - Minimum [s]
  - Mean [s]
* - automl_conf_0
  - 0.000010
  - 0.001173
  - 0.001074
  - 0.001042
  - 0.001075
* - automl_conf_1
  - 0.000005
  - 0.001101
  - 0.001074
  - 0.001053
  - 0.001074
* - automl_conf_3
  - 0.000005
  - 0.000958
  - 0.000945
  - 0.000916
  - 0.000946
* - automl_conf_4
  - 0.000004
  - 0.000421
  - 0.000407
  - 0.000390
  - 0.000407
* - automl_conf_5
  - 0.000004
  - 0.000444
  - 0.000408
  - 0.000395
  - 0.000407


```










### Mean comparison plots

```{figure} generated/img/mean_performance_comparison.*
---
name: classification_and_reporttypesperformance_of_automl_conf_0_performance_comparison
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
  - 0.001075
* - automl_conf_1
  - 0.001074
* - automl_conf_3
  - 0.000946
* - automl_conf_4
  - 0.000407
* - automl_conf_5
  - 0.000407
```

