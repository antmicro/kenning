## Depth estimation metrics{% if data["model_name"] %} for {{data["model_name"]}}{% endif %}

{% set basename = data["report_name_simple"] if "model_name" not in data else data["report_name_simple"] + data["model_name"] %}

```{list-table} Depth estimation metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Value
{%- for metric in data["metrics"] %}
* - {{metric.value or metric}}
  - {{data[metric]}}
{%- endfor %}
```

{%- for sample_category in data["sample_categories"] %}
{% if data[sample_category + "_plot_paths"] %}
### {{ sample_category.capitalize() }} sample predictions
{% endif %}

{%- for plot_path in data[sample_category + "_plot_paths"] %}
```{image} {{plot_path}}
:alt: Depth estimation sample prediction
:class: bg-primary
:width: 50%
:align: center
```


{%- endfor %}
{%- endfor %}

