## Classification comparison

### Comparison of inference time, accuracy and model size

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

### Detailed metrics comparison

```{figure} {{data['radarchartpath']}}
---
name: {{data['report_name_simple']}}_classification_metrics_radar
alt: Metric comparison
align: center
width: 100%
---

Radar chart representing the accuracy, precision and recall for models
```

```{list-table} Summary of classification metrics for models
---
header-rows: 1
align: center
---

* - Model name
{%- for metric in data["available_metrics"] %}
  - {{metric.value}}
{%- endfor %}
{% for model_name in data["model_names"] %}
* - {{model_name}}
{%- for metric in data["available_metrics"] %}
  - {{'**' if data[model_name][metric] == data['max_metrics'][metric]}}{{'%.6f' % data[model_name][metric]}}{{'**' if data[model_name][metric] == data['max_metrics'][metric]}}
{%- endfor %}
{% endfor %}
```

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


