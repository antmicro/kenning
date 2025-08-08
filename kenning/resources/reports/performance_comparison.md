## Inference comparison

### Performance metrics

{% macro displayMetrics(name, unit, prec) %}
* - Model name {{"\n"}}
{%- for metric in data["available_metrics"] -%}
{%- if name in metric -%}
{%- if 'mean' in metric %}  - Mean  {%- endif -%}
{%- if 'median' in metric %}  - Median {%- endif -%}
{%- if 'std' in metric %}  - Standard deviation {%- endif -%}
{%- if 'min' in metric %}  - Minimum {%- endif -%}
{%- if 'max' in metric %}  - Maximum {%- endif -%}
{{" ["}}{{unit}}{{"]\n"}}
{%- endif -%}
{%- endfor -%}
{%- for model_name in data["model_names"] -%}
* - {{model_name}} {{"\n"}}
{%- for metric in data["available_metrics"] -%}
{%- if name in metric %}  - {{prec % data[model_name][metric]}}{{"\n"}} {%- endif -%}
{%- endfor -%}
{%- endfor -%}
{% endmacro %}

{% if 'inference_step_path' in data -%}
```{figure} {{data["inference_step_path"]}}
---
name: {{data["report_name_simple"]}}_inference_step_comparison
alt: Inference time comparison
align: center
---

Plot represents changes of inference time over time for all models.
```

```{list-table} Summary of inference time metrics for models
---
header-rows: 1
align: center
---

{{ displayMetrics('inferencetime', 's', '%.6f') }}

```
{% endif %}

{% if 'session_utilization_cpus_percent_avg_path' in data -%}
```{figure} {{data["session_utilization_cpus_percent_avg_path"]}}
---
name: {{data["report_name_simple"]}}_cpu_comparison
alt: CPU usage comparison
align: center
---

Plot represents changes of CPU usage over time for all models.
```

```{list-table} Summary of CPU usage metrics for models
---
header-rows: 1
align: center
---

{{ displayMetrics('cpus_percent_avg', '%', '%.3f') }}
```
{% endif %}

{% if 'session_utilization_mem_percent_path' in data -%}
```{figure} {{data["session_utilization_mem_percent_path"]}}
---
name: {{data["report_name_simple"]}}_memory_comparison
alt: Memory usage comparison
align: center
---

Plot represents changes of RAM usage over time for all models.
```
```{list-table} Summary of RAM usage metrics for models
---
header-rows: 1
align: center
---

{{ displayMetrics('utilization_mem_percent', '%', '%.3f') }}
```
{% endif %}

{% if 'session_utilization_gpu_utilization_path' in data -%}
```{figure} {{data["session_utilization_gpu_utilization_path"]}}
---
name: {{data["report_name_simple"]}}_gpu_comparison
alt: GPU usage comparison
align: center
---

Plot represents changes of GPU usage over time for all models.
```
```{list-table} Summary of GPU usage metrics for models
---
header-rows: 1
align: center
---

{{ displayMetrics('gpu_utilization', '%', '%.3f') }}
```
{% endif %}

{% if 'session_utilization_gpu_mem_utilization_path' in data -%}
```{figure} {{data["session_utilization_gpu_mem_utilization_path"]}}
---
name: {{data["report_name_simple"]}}_gpu_mem_comparison
alt: GPU memory usage comparison
align: center
---

Plot represents changes of GPU RAM usage over time for all models.
```
```{list-table} Summary of GPU RAM usage metrics for models
---
header-rows: 1
align: center
---

{{ displayMetrics('gpu_mem', 'MB', '%.6f') }}
```
{% endif %}

### Mean comparison plots

```{figure} {{data["meanperformancepath"]}}
---
name: {{data["report_name_simple"]}}_performance_comparison
alt: Performance comparison
align: center
---
Violin chart representing distribution of values for performance metrics for models
```
{% macro displayStatsM(name, unit, prec, model_name) %}
    ```{list-table}
    ---
    header-rows: 0
    align: center
    ---
{%- for metric in data["available_metrics"] %}
{%- if name in metric -%} {{"\n    * - "}}
{%- if 'mean' in metric %}Mean  {%- endif -%}
{%- if 'median' in metric %}Median {%- endif -%}
{%- if 'std' in metric %}Standard deviation {%- endif -%}
{%- if 'min' in metric %}Minimum {%- endif -%}
{%- if 'max' in metric %}Maximum {%- endif -%}
{{" ["}}{{unit}}{{"]\n      - "}}{{prec % data[model_name][metric]}}{{"\n"}}
{%- endif -%}
{% endfor %}
    ```
{% endmacro %}

```{list-table} Performance metric for models
---
header-rows: 1
align: center
---
* - Model name
  - Inference time
  - CPU usage
  - Memory usage
{% for model_name in data["model_names"] -%}
* - {{model_name}}
  -
{{- displayStatsM('inferencetime', 's', '%.6f', model_name) -}}
{{"  -"}}
{{- displayStatsM('cpus_percent_avg', '%', '%.3f', model_name) -}}
{{"  -"}}
{{- displayStatsM('utilization_mem_percent', '%', '%.3f', model_name) -}}

{%- endfor -%}
```

{%- if 'hardwareusagepath' in data %}
### Hardware usage comparison

```{figure} {{data["hardwareusagepath"]}}
---
name: {{data["report_name_simple"]}}_hardware_usage_comparison
alt: Resource usage comparison
align: center
---
Radar chart representing the resource usage of models
```
```{list-table} Hardware usage statistics
---
header-rows: 0
align: center
---
* - Model name
  - Average CPU usage [%]
  - Average memory usage [%]
{% if 'session_utilization_gpu_mem_utilization_mean' in data[model_name] %}
  - Average GPU usage [%]
  - Average GPU memory usage [MB]
{% endif %}
{% for model_name in data["model_names"] %}
* - {{model_name}}
  - {{'%.3f' % data[model_name]["session_utilization_cpus_percent_avg_mean"]}}
  - {{'%.3f' % data[model_name]["session_utilization_mem_percent_mean"]}}
{% if 'session_utilization_gpu_mem_utilization_mean' in data[model_name] %}
  - {{'%.3f' % data[model_name]["session_utilization_gpu_utilization_mean"]}}
  - {{'%.6f' % data[model_name]["session_utilization_gpu_mem_utilization_mean"]}}
{% endif %}
{% endfor %}
```
{%- endif %}


