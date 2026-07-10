## Depth estimation metrics{% if data["model_name"] %} for {{data["model_name"]}}{% endif %}

{% set basename = data["report_name_simple"] if "model_name" not in data else data["report_name_simple"] + data["model_name"] %}

```{list-table} Depth estimation metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Value
{%- for metric in data %}
* - {{metric.value or metric}}
  - {{data[metric]}}
{%- endfor %}
```

