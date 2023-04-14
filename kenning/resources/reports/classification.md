## Inference quality metrics{% if data["modelname"] %} for {{data["modelname"]}}{% endif %}

{% set basename = data["reportname_simple"] if "modelname" not in data else data["reportname_simple"] + data["modelname"] %}
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

