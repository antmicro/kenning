## Inference performance in cycles{% if data["model_name"] %} for {{data["model_name"]}}{% endif %}

{% set basename = data["report_name_simple"] if "model_name" not in data else data["report_name_simple"] + data["model_name"] %}
{% if 'cycles_series_plot' in data or 'mean_cycles' in data or 'total_cycles' in data -%}
### Cycles usage across inference

{%- if 'cycles_series_plot' in data %}

```{figure} {{data["cycles_series_plot"]}}
---
name: {{basename}}_cycles_series_plot
alt: Cycles usage across inference
align: center
---

Cycles usage across inference
```
{% endif %}


{%- if 'mean_cycles' in data %}
Mean number of cycles per one inference: **{{ data['mean_cycles'] }}**
{% endif %}
{%- if 'total_cycles' in data %}
Total number of cycles: **{{ data['total_cycles'] }}**
{% endif %}

{% endif %}
