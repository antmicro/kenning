## Regression metrics{% if data["model_name"] %} for {{data["model_name"]}}{% endif %}

{% set basename = data["report_name_simple"] if "model_name" not in data else data["report_name_simple"] + data["model_name"] %}

```{list-table} Regression metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Value
{%- for metric in data %}
* - {{metric.value}}
  - {{data[metric]}}
{%- endfor %}
```

