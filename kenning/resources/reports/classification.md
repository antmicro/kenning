## Inference quality metrics{% if data["model_name"] %} for {{data["model_name"]}}{% endif %}

{% set basename = data["report_name_simple"] if "model_name" not in data else data["report_name_simple"] + data["model_name"] %}
```{figure} {{data["confusionpath"]}}
---
name: {{basename}}_confusionmatrix
alt: Confusion matrix
align: center
---

Confusion matrix
```

* *Accuracy*: **{{ data['accuracy'] }}**
{%- if data['top_5_accuracy'] %}
* *Top-5 accuracy*: **{{ data['top_5_accuracy'] }}**
{%- endif %}
* *Mean precision*: **{{ data['mean_precision'] }}**
* *Mean sensitivity*: **{{ data['mean_sensitivity'] }}**
* *G-mean*: **{{ data['g_mean'] }}**

