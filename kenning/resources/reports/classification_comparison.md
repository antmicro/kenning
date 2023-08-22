## Classification comparison

{%- if 'predictionsbarpath' in data %}
```{figure} {{data["predictionsbarpath"]}}
---
name: {{basename}}_predictionsbarplot
alt: Predictions comparison
align: center
---

Predictions comparison
```
{%- endif %}

```{figure} {{data["bubbleplotpath"]}}
---
name: {{data["report_name_simple"]}}_classification_size_inference
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
name: {{data['report_name_simple']}}_classification_metrics_radar
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
  - Mean precision
  - Mean recall
{% for model_name in data["model_names"] %}
* - {{model_name}}
  - {{'%.6f' % data[model_name][0]}}
  - {{'%.6f' % data[model_name][1]}}
  - {{'%.6f' % data[model_name][2]}}
{% endfor %}
```


