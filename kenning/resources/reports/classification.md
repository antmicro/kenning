## Inference quality metrics{% if data["model_name"] %} for {{data["model_name"]}}{% endif %}

{% set basename = data["report_name_simple"] if "model_name" not in data else data["report_name_simple"] + data["model_name"] %}

{%- if 'confusionpath' in data %}
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
{%- if data['f1_score'] %}
* *F1 score*: **{{ data['f1_score'] }}**
{%- endif %}
{%- if data['f1_score_weighted'] %}
* *weighted F1 score*: **{{ data['f1_score_weighted'] }}**
{%- for class_, score in zip(data['class_names'], data['f1_score_per_class']) %}
  * *F1 score for {{ class_ }}*: **{{ score }}**
{%- endfor %}
{%- endif %}
{%- if data['roc_auc'] %}
* *ROC AUC*: **{{ data['roc_auc'] }}**
{%- endif %}
{%- if data['roc_auc_weighted'] %}
* *weighted ROC AUC*: **{{ data['roc_auc_weighted'] }}**
{%- for class_, score in zip(data['class_names'], data['roc_auc_per_class']) %}
  * *ROC AUC for {{ class_ }}*: **{{ score }}**
{%- endfor %}
{%- endif %}

{%- endif %}

{%- if 'predictionspath' in data %}
```{figure} {{data["predictionspath"]}}
---
name: {{basename}}_predictionsbarplot
alt: Predictions
align: center
---

Predictions
```
{%- endif %}

