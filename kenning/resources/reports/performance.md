## Inference performance metrics{% if data["model_name"] %} for {{data["model_name"]}}{% endif %}

{% set basename = data["report_name_simple"] if "model_name" not in data else data["report_name_simple"] + data["model_name"] %}
{% if 'inferencetime' in data -%}
### Inference time

```{figure} {{data["inferencetimepath"]}}
---
name: {{basename}}_inferencetime
alt: Inference time
align: center
---

Inference time
```
* *First inference duration* (usually including allocation time): **{{ data['inferencetime'][0] }}**,
* *Mean*: **{{ data['inferencetime_mean'] }} s**,
* *Standard deviation*: **{{ data['inferencetime_std'] }} s**,
* *Median*: **{{ data['inferencetime_median'] }} s**.
{% endif %}

{% if 'session_utilization_cpus_percent_avg' in data -%}
### Average CPU usage

```{figure} {{data["cpuusagepath"]}}
---
name: {{basename}}_cpuusage
alt: Average CPU usage
align: center
---

Average CPU usage during benchmark
```

* *Mean*: **{{ data['session_utilization_cpus_percent_avg_mean'] }} %**,
* *Standard deviation*: **{{ data['session_utilization_cpus_percent_avg_std'] }} %**,
* *Median*: **{{ data['session_utilization_cpus_percent_avg_median'] }} %**.
{% endif %}

{% if 'session_utilization_mem_percent' in data -%}
### Memory usage

```{figure} {{data["memusagepath"]}}
---
name: {{basename}}_memusage
alt: Memory usage
align: center
---

Memory usage during benchmark
```

* *Mean*: **{{ data['session_utilization_mem_percent_mean'] }} %**,
* *Standard deviation*: **{{ data['session_utilization_mem_percent_std'] }} %**,
* *Median*: **{{ data['session_utilization_mem_percent_median'] }} %**.
{% endif %}

{% if 'session_utilization_gpu_utilization' in data and data['session_utilization_gpu_utilization']|length > 0 -%}
## GPU usage


```{figure} {{data["gpuusagepath"]}}
---
name: {{basename}}_gpuusage
alt: GPU usage
align: center
---

GPU utilization during benchmark
```

* *Mean*: **{{ data['session_utilization_gpu_utilization_mean'] }} %**,
* *Standard deviation*: **{{ data['session_utilization_gpu_utilization_std'] }} %**,
* *Median*: **{{ data['session_utilization_gpu_utilization_median'] }} %**.
{% endif %}

{% if 'session_utilization_gpu_mem_utilization' in data and data['session_utilization_gpu_mem_utilization']|length > 0 -%}
## GPU memory usage

```{figure} {{data["gpumemusagepath"]}}
---
name: {{basename}}_gpumemusage
alt: GPU memory usage
align: center
---

GPU memory usage during benchmark
```

* *Mean*: **{{ data['session_utilization_gpu_mem_utilization_mean'] }} MB**,
* *Standard deviation*: **{{ data['session_utilization_gpu_mem_utilization_std'] }} MB**,
* *Median*: **{{ data['session_utilization_gpu_mem_utilization_median'] }} MB**.
{% endif %}

