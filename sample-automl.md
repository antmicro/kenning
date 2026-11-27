## AutoML statistics


- *Optimized metric*: **f1**

- *The number of generated models*: **37**

- *The number of trained and evaluated models*: **25**

- *The number of successful training processes*: **31**

- *The number of models that caused a crash*: **2**

- *The number of models that failed due to the timeout*: **1**

- *The number of models that failed due to the too large size*: **3**

- *The number of models that failed due to incompatibility*: **0**



### Training overview

```{figure} generated/img/sample-automl.md/training_plot.*
---
name: automl_training_plot
alt: Loss value during AutoML training process
align: center
---

Loss value during AutoML training process
```

```{figure} generated/img/sample-automl.md/comparison_training_plot.*
---
name: automl_comparison_training_plot
alt: Comparison of loss value across models
align: center
---

Comparison of loss value across models
```



### Summary of generated models




```{figure} generated/img/sample-automl.md/trained_models_plot.*
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
| 3 |  7 | 15.55078125 | 2815 | 2814 |
| 4 |  10 | 49.71484375 | 11623 | 11622 |
| 5 |  17 | 33.0859375 | 7498 | 7497 |
| 6 |  21 | 33.49609375 | 7613 | 7612 |
| 7 |  27 | 61.03125 | 14094 | 14093 |
| 8 |  14 | 50.1640625 | 11841 | 11840 |
| 9 |  21 | 35.58984375 | 7834 | 7833 |
| 10 |  17 | 37.43359375 | 8732 | 8731 |
| 11 |  23 | 54.3203125 | 11691 | 11690 |
| 12 |  21 | 30.3828125 | 7656 | 7655 |
| 13 |  17 | 33.140625 | 7799 | 7798 |
| 14 |  12 | 37.56640625 | 8227 | 8226 |
| 15 |  10 | 42.42578125 | 10264 | 10263 |
| 16 |  13 | 45.5390625 | 10455 | 10454 |
| 17 |  8 | 19.87109375 | 4364 | 4363 |
| 18 |  27 | 47.51171875 | 10720 | 10719 |
| 19 |  13 | 26.8515625 | 7493 | 7492 |
| 20 |  10 | 26.984375 | 5881 | 5880 |
| 21 |  23 | 42.40625 | 10122 | 10121 |
| 22 |  25 | 45.64453125 | 10132 | 10131 |
| 23 |  17 | 28.19921875 | 6185 | 6184 |
| 24 |  10 | 44.74609375 | 10414 | 10413 |
| 25 |  11 | 30.41015625 | 6877 | 6876 |
| 26 |  21 | 25.765625 | 5136 | 5135 |
| 27 |  14 | 29.5390625 | 5655 | 5654 |
| 28 |  11 | 55.3515625 | 14297 | 14296 |
| 29 |  13 | 29.1171875 | 5933 | 5932 |
| 30 |  23 | 33.078125 | 7653 | 7652 |
| 31 |  13 | 33.1484375 | 7842 | 7841 |
| 32 |  13 | 23.1171875 | 4937 | 4936 |
| 33 |  11 |  | 9790 | 9789 |

```

## Classification comparison

### Comparison of inference time, F1 score and model size

```{figure} generated/img/sample-automl.md/f1_vs_inference_time.*
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
  - 0.000852
  - 0.034
  - 0.533333

* - automl_conf_1
  - 0.000426
  - 0.016
  - 0.533333

* - automl_conf_2
  - 0.000418
  - 0.016
  - 0.533333

* - automl_conf_3
  - 0.001228
  - 0.047
  - 0.470588

* - automl_conf_4
  - 0.000526
  - 0.024
  - 0.533333

```

### Detailed metrics comparison

```{figure} generated/img/sample-automl.md/classification_metric_comparison.*
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

* - automl_conf_2
  - **0.972000**
  - **0.777484**
  - **0.743802**
  - **0.702710**
  - **0.743802**
  - **0.533333**

* - automl_conf_3
  - 0.964000
  - 0.713923
  - 0.739669
  - 0.699764
  - 0.739669
  - 0.470588

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



```{figure} generated/img/sample-automl.md/inference_step_comparison.*
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
  - Maximum [s]
  - Mean [s]
  - Median [s]
  - Minimum [s]
  - Standard deviation [s]
* - automl_conf_0
  - 0.001021
  - 0.000852
  - 0.000832
  - 0.000794
  - 0.000042
* - automl_conf_1
  - 0.000601
  - 0.000426
  - 0.000409
  - 0.000387
  - 0.000037
* - automl_conf_2
  - 0.000602
  - 0.000418
  - 0.000406
  - 0.000387
  - 0.000032
* - automl_conf_3
  - 0.001497
  - 0.001228
  - 0.001213
  - 0.001191
  - 0.000038
* - automl_conf_4
  - 0.000925
  - 0.000526
  - 0.000521
  - 0.000507
  - 0.000029


```










### Mean comparison plots

```{figure} generated/img/sample-automl.md/mean_performance_comparison.*
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
  - 0.000852
* - automl_conf_1
  - 0.000426
* - automl_conf_2
  - 0.000418
* - automl_conf_3
  - 0.001228
* - automl_conf_4
  - 0.000526
```

