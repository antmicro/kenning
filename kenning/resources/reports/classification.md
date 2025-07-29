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

```{list-table} Inference quality metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Value
{%- for metric in data['available_metrics'] %}
{%- if metric.name.endswith("_CLASS") %}
{%- for class_, score in zip(data['class_names'], data[metric]) %}
* - *{{metric.value}} for {{ class_ }}*
  - **{{'%.6f' % score }}**
{%- endfor %}
{%- else %}
* - *{{ metric.value }}*
  - **{{'%.6f' %  data[metric] }}**
{%- endif %}
{%- endfor %}
```

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

