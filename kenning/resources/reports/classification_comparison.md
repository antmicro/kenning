## Classification comparison

```{figure} {{data["bubbleplotpath"]}}
---
name: {{data["reportname_simple"]}}_classification_size_inference
alt: Accuracy vs Inference time vs RAM usage
align: center
---

Model size, speed and quality summary.
The accuracy of the model is presented on Y axis.
The inference time of the model is presented on X axis.
The size of the model is represented by the size of its point.
```

### Metric comparison

```{figure} {{data['radarchartpath']}}
---
name: {{data['reportname_simple']}}_classification_metrics_radar
alt: Metric comparison
align: center
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
  - Mean precison
  - Mean recall
{% for modelname in data["modelnames"] %}
* - {{modelname}}
  - {{'%.6f' % data[modelname][0]}}
  - {{'%.6f' % data[modelname][1]}}
  - {{'%.6f' % data[modelname][2]}}
{% endfor %}
```


