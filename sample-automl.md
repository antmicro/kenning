## AutoML statistics


- *Optimized metric*: **f1**

- *The number of generated models*: **48**

- *The number of trained and evaluated models*: **31**

- *The number of successful training processes*: **39**

- *The number of models that caused a crash*: **0**

- *The number of models that failed due to the timeout*: **1**

- *The number of models that failed due to the too large size*: **8**

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
| 19 |  13 | 29.05859375 | 5596 | 5595 |
| 20 |  25 | 37.98046875 | 7352 | 7351 |
| 21 |  13 | 37.41796875 | 7850 | 7849 |
| 22 |  13 | 44.48828125 | 9072 | 9071 |
| 23 |  14 | 50.4609375 | 10981 | 10980 |
| 24 |  27 | 40.328125 | 8374 | 8373 |
| 25 |  11 | 22.15625 | 3781 | 3780 |
| 26 |  13 | 20.4375 | 3701 | 3700 |
| 27 |  21 | 34.546875 | 6551 | 6550 |
| 28 |  19 | 61.23046875 | 13954 | 13953 |
| 29 |  13 | 52.6328125 | 11737 | 11736 |
| 30 |  10 | 36.63671875 | 7190 | 7189 |
| 31 |  11 | 61.73046875 | 12885 | 12884 |
| 32 |  17 | 37.41015625 | 7738 | 7737 |
| 33 |  8 | 47.0703125 | 10540 | 10539 |
| 34 |  10 | 33.56640625 | 6402 | 6401 |
| 35 |  19 | 49.11328125 | 9859 | 9858 |
| 36 |  15 | 28.00390625 | 5337 | 5336 |
| 37 |  10 | 29.0390625 | 5245 | 5244 |
| 38 |  8 | 29.05859375 | 5444 | 5443 |
| 39 |  10 | 30.578125 | 5748 | 5747 |
| 40 |  17 | 56.97265625 | 12815 | 12814 |
| 41 |  12 | 47.671875 | 9095 | 9094 |
| 42 |  9 |  | 6448 | 6447 |

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
  - 0.000556
  - 0.018
  - 0.250000

* - workspace.automl-results.1234_12_5.0.measurements.json
  - 0.001246
  - 0.040
  - 0.250000

* - workspace.automl-results.1234_27_5.0.measurements.json
  - 0.001236
  - 0.036
  - 0.352941

* - workspace.automl-results.1234_30_1.6666666666666665.measurements.json
  - 0.002890
  - 0.038
  - 0.000000

* - workspace.automl-results.1234_30_5.0.measurements.json
  - 0.002889
  - 0.038
  - 0.000000

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

* - workspace.automl-results.1234_27_5.0.measurements.json
  - **0.956000**
  - **0.781633**
  - **0.620798**
  - **0.497895**
  - **0.620798**
  - **0.352941**

* - workspace.automl-results.1234_30_1.6666666666666665.measurements.json
  - 0.952000
  - 0.476000
  - 0.500000
  - 0.000000
  - 0.500000
  - 0.000000

* - workspace.automl-results.1234_30_5.0.measurements.json
  - 0.952000
  - 0.476000
  - 0.500000
  - 0.000000
  - 0.500000
  - 0.000000

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
  - Median [s]
  - Minimum [s]
  - Standard deviation [s]
  - Maximum [s]
* - workspace.automl-results.1234_3_5.0.measurements.json
  - 0.000556
  - 0.000550
  - 0.000525
  - 0.000022
  - 0.000666
* - workspace.automl-results.1234_12_5.0.measurements.json
  - 0.001246
  - 0.001240
  - 0.001206
  - 0.000021
  - 0.001341
* - workspace.automl-results.1234_27_5.0.measurements.json
  - 0.001236
  - 0.001235
  - 0.001187
  - 0.000021
  - 0.001323
* - workspace.automl-results.1234_30_1.6666666666666665.measurements.json
  - 0.002890
  - 0.002884
  - 0.002478
  - 0.000153
  - 0.003290
* - workspace.automl-results.1234_30_5.0.measurements.json
  - 0.002889
  - 0.002890
  - 0.002455
  - 0.000153
  - 0.003298


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
  - 0.000556
* - workspace.automl-results.1234_12_5.0.measurements.json
  - 0.001246
* - workspace.automl-results.1234_27_5.0.measurements.json
  - 0.001236
* - workspace.automl-results.1234_30_1.6666666666666665.measurements.json
  - 0.002890
* - workspace.automl-results.1234_30_5.0.measurements.json
  - 0.002889
```

