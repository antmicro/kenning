## AutoML statistics


- *Optimized metric*: **f1**

- *The number of generated models*: **42**

- *The number of trained and evaluated models*: **29**

- *The number of successful training processes*: **36**

- *The number of models that caused a crash*: **2**

- *The number of models that failed due to the timeout*: **1**

- *The number of models that failed due to the too large size*: **3**

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
| 3 |  7 | 15.5546875 | 2815 | 2814 |
| 4 |  10 | 49.73046875 | 11623 | 11622 |
| 5 |  17 | 33.08203125 | 7498 | 7497 |
| 6 |  21 | 33.4921875 | 7613 | 7612 |
| 7 |  27 | 61.02734375 | 14094 | 14093 |
| 8 |  14 | 50.16015625 | 11841 | 11840 |
| 9 |  21 | 35.60546875 | 7834 | 7833 |
| 10 |  17 | 37.4296875 | 8732 | 8731 |
| 11 |  23 | 54.3359375 | 11691 | 11690 |
| 12 |  21 | 30.37890625 | 7656 | 7655 |
| 13 |  17 | 33.13671875 | 7799 | 7798 |
| 14 |  12 | 37.5625 | 8227 | 8226 |
| 15 |  10 | 42.421875 | 10264 | 10263 |
| 16 |  13 | 45.5546875 | 10455 | 10454 |
| 17 |  8 | 19.8671875 | 4364 | 4363 |
| 18 |  27 | 47.5078125 | 10720 | 10719 |
| 19 |  25 | 49.37109375 | 13209 | 13208 |
| 20 |  19 | 45.875 | 10707 | 10706 |
| 21 |  12 | 44.40234375 | 10171 | 10170 |
| 22 |  14 | 58.640625 | 14009 | 14008 |
| 23 |  9 | 27.69921875 | 6301 | 6300 |
| 24 |  10 | 24.9453125 | 5151 | 5150 |
| 25 |  9 | 50.67578125 | 12130 | 12129 |
| 26 |  25 | 51.265625 | 11652 | 11651 |
| 27 |  11 | 30.75390625 | 8095 | 8094 |
| 28 |  10 | 36.19140625 | 8007 | 8006 |
| 29 |  21 | 24.62890625 | 4788 | 4787 |
| 30 |  23 | 30.2890625 | 6668 | 6667 |
| 31 |  13 | 26.79296875 | 6005 | 6004 |
| 32 |  19 | 41.4453125 | 8872 | 8871 |
| 33 |  19 | 42.16015625 | 9754 | 9753 |
| 34 |  15 | 31.8359375 | 6497 | 6496 |
| 35 |  11 | 42.99609375 | 9219 | 9218 |
| 36 |  13 | 20.0625 | 4112 | 4111 |
| 37 |  15 | 31.16796875 | 6868 | 6867 |

```

## Classification comparison

### Comparison of inference time, F1 score and model size

```{figure} generated/img/accuracy_vs_inference_time.*
---
name: classification_and_reporttypesperformance_of_workspaceautomlresults1234_3_50measurementsjson_classification_size_inference
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
  - 0.000421
  - 0.016
  - 0.250000

* - workspace.automl-results.1234_12_5.0.measurements.json
  - 0.000857
  - 0.032
  - 0.250000

* - workspace.automl-results.1234_19_1.6666666666666665.measurements.json
  - 0.004029
  - 0.052
  - 0.352941

* - workspace.automl-results.1234_19_5.0.measurements.json
  - 0.004028
  - 0.052
  - 0.352941

* - workspace.automl-results.1234_31_5.0.measurements.json
  - 0.000739
  - 0.028
  - 0.250000

```

### Detailed metrics comparison

```{figure} generated/img/classification_metric_comparison.*
---
name: classification_and_reporttypesperformance_of_workspaceautomlresults1234_3_50measurementsjson_classification_metrics_radar
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

* - workspace.automl-results.1234_19_1.6666666666666665.measurements.json
  - **0.956000**
  - **0.781633**
  - **0.620798**
  - **0.497895**
  - **0.620798**
  - **0.352941**

* - workspace.automl-results.1234_19_5.0.measurements.json
  - **0.956000**
  - **0.781633**
  - **0.620798**
  - **0.497895**
  - **0.620798**
  - **0.352941**

* - workspace.automl-results.1234_31_5.0.measurements.json
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
name: classification_and_reporttypesperformance_of_workspaceautomlresults1234_3_50measurementsjson_inference_step_comparison
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
  - Mean [s]
  - Minimum [s]
  - Standard deviation [s]
  - Median [s]
  - Maximum [s]
* - workspace.automl-results.1234_3_5.0.measurements.json
  - 0.000421
  - 0.000379
  - 0.000022
  - 0.000416
  - 0.000517
* - workspace.automl-results.1234_12_5.0.measurements.json
  - 0.000857
  - 0.000810
  - 0.000033
  - 0.000841
  - 0.000940
* - workspace.automl-results.1234_19_1.6666666666666665.measurements.json
  - 0.004029
  - 0.003638
  - 0.000180
  - 0.004026
  - 0.004545
* - workspace.automl-results.1234_19_5.0.measurements.json
  - 0.004028
  - 0.003622
  - 0.000178
  - 0.004014
  - 0.004543
* - workspace.automl-results.1234_31_5.0.measurements.json
  - 0.000739
  - 0.000697
  - 0.000022
  - 0.000733
  - 0.000833


```










### Mean comparison plots

```{figure} generated/img/mean_performance_comparison.*
---
name: classification_and_reporttypesperformance_of_workspaceautomlresults1234_3_50measurementsjson_performance_comparison
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
  - 0.000421
* - workspace.automl-results.1234_12_5.0.measurements.json
  - 0.000857
* - workspace.automl-results.1234_19_1.6666666666666665.measurements.json
  - 0.004029
* - workspace.automl-results.1234_19_5.0.measurements.json
  - 0.004028
* - workspace.automl-results.1234_31_5.0.measurements.json
  - 0.000739
```

