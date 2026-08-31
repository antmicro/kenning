## AutoML statistics


- *Optimized metric*: **f1**

- *The number of generated models*: **52**

- *The number of trained and evaluated models*: **39**

- *The number of successful training processes*: **47**

- *The number of models that caused a crash*: **0**

- *The number of models that failed due to the timeout*: **1**

- *The number of models that failed due to the too large size*: **4**

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
| 24 |  19 | 45.32421875 | 10190 | 10189 |
| 25 |  11 | 30.05859375 | 6573 | 6572 |
| 26 |  25 | 57.24609375 | 13239 | 13238 |
| 27 |  13 | 23.9140625 | 5428 | 5427 |
| 28 |  11 | 30.703125 | 7744 | 7743 |
| 29 |  13 | 52.5625 | 13700 | 13699 |
| 30 |  21 | 32.3671875 | 7593 | 7592 |
| 31 |  11 | 44.21875 | 9580 | 9579 |
| 32 |  9 | 22.546875 | 5868 | 5867 |
| 33 |  17 | 50.02734375 | 12217 | 12216 |
| 34 |  12 | 51.5546875 | 13437 | 13436 |
| 35 |  19 | 31.24609375 | 6620 | 6619 |
| 36 |  12 | 40.328125 | 8949 | 8948 |
| 37 |  17 | 36.046875 | 7788 | 7787 |
| 38 |  19 | 47.15625 | 11050 | 11049 |
| 39 |  13 | 28.4375 | 6345 | 6344 |
| 40 |  11 | 42.58984375 | 10169 | 10168 |
| 41 |  9 | 39.98828125 | 9895 | 9894 |
| 42 |  9 | 12.5859375 | 2351 | 2350 |
| 43 |  17 | 20.21875 | 3853 | 3852 |
| 44 |  21 | 42.87890625 | 9632 | 9631 |
| 45 |  15 | 45.53125 | 11264 | 11263 |

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
  - 0.003192
  - 0.043
  - 0.571429

* - automl_conf_1
  - 0.000840
  - 0.032
  - 0.571429

* - automl_conf_2
  - 0.000839
  - 0.032
  - 0.571429

* - automl_conf_3
  - 0.000951
  - 0.037
  - 0.533333

* - automl_conf_4
  - 0.000414
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
  - **0.976000**
  - **0.825137**
  - **0.745868**
  - **0.704179**
  - **0.745868**
  - **0.571429**

* - automl_conf_1
  - **0.976000**
  - **0.825137**
  - **0.745868**
  - **0.704179**
  - **0.745868**
  - **0.571429**

* - automl_conf_2
  - **0.976000**
  - **0.825137**
  - **0.745868**
  - **0.704179**
  - **0.745868**
  - **0.571429**

* - automl_conf_3
  - 0.972000
  - 0.777484
  - 0.743802
  - 0.702710
  - 0.743802
  - 0.533333

* - automl_conf_4
  - 0.972000
  - 0.777484
  - 0.743802
  - 0.702710
  - 0.743802
  - 0.533333

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
  - Median [s]
  - Minimum [s]
  - Maximum [s]
  - Mean [s]
* - automl_conf_0
  - 0.000139
  - 0.003194
  - 0.002815
  - 0.003527
  - 0.003192
* - automl_conf_1
  - 0.000015
  - 0.000836
  - 0.000801
  - 0.000901
  - 0.000840
* - automl_conf_2
  - 0.000015
  - 0.000835
  - 0.000803
  - 0.000900
  - 0.000839
* - automl_conf_3
  - 0.000014
  - 0.000948
  - 0.000917
  - 0.001017
  - 0.000951
* - automl_conf_4
  - 0.000010
  - 0.000411
  - 0.000392
  - 0.000476
  - 0.000414


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
  - 0.003192
* - automl_conf_1
  - 0.000840
* - automl_conf_2
  - 0.000839
* - automl_conf_3
  - 0.000951
* - automl_conf_4
  - 0.000414
```

