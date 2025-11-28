## Tokens per second comparison

{% set basename = data["report_name_simple"] if "model_name" not in data else data["report_name_simple"] + data["model_name"] %}

### Tokens per second
```{figure} {{data["barplot_tokens_per_second_comparison"]}}

---
name: {{basename}}_tokens_per_second
alt: Tokens per second
align: center
---
Tokens per second
```

```{list-table} Summary of tokens per second metrics for models
---
header-rows: 1
align: center
---

  * - Model name
    - Mean
    - Standard deviation
    - Median
    - Minimum
    - Maximum
{% for model_name in data["model_names"] %}
  * - {{model_name}}
    - **{{'%.3f' %  data[model_name]['tokens_per_second_mean'] }}**
    - **{{'%.3f' %  data[model_name]['tokens_per_second_std'] }}**
    - **{{'%.3f' %  data[model_name]['tokens_per_second_median'] }}**
    - **{{'%.3f' %  data[model_name]['tokens_per_second_min'] }}**
    - **{{'%.3f' %  data[model_name]['tokens_per_second_max'] }}**
{% endfor %}

```


