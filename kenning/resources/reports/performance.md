## Inference performance metrics{% if data["modelname"] %} for {{data["modelname"]}}{% endif %}

{% set basename = data["reportname_simple"] if "modelname" not in data else data["reportname_simple"] + data["modelname"] %}
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
### Mean CPU usage

```{figure} {{data["cpuusagepath"]}}
---
name: {{basename}}_cpuusage
alt: Mean CPU usage
align: center
---

Mean CPU usage during benchmark
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

