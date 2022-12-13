# Inference performance metrics{% if data["modelname"] %} for {{data["modelname"]}}{% endif %}

{% set basename = data["reportname"] if "modelname" not in data else data["reportname"] + data["modelname"] %}
{% if 'inferencetime' in data -%}
## Inference time

```{figure} {{data["inferencetimepath"]}}
---
name: {{basename}}_inferencetime
alt: Inference time
align: center
---

Inference time
```

* *First inference duration* (usually including allocation time): **{{ data['inferencetime'][0] }}**,
* *Mean*: **{{ mean(data['inferencetime']) }} s**,
* *Standard deviation*: **{{ std(data['inferencetime']) }} s**,
* *Median*: **{{ median(data['inferencetime']) }} s**.
{% endif %}

{% if 'session_utilization_cpus_percent_avg' in data -%}
## Mean CPU usage

```{figure} {{data["cpuusagepath"]}}
---
name: {{basename}}_cpuusage
alt: Mean CPU usage
align: center
---

Mean CPU usage during benchmark
```

* *Mean*: **{{ mean(data['session_utilization_cpus_percent_avg']) }} %**,
* *Standard deviation*: **{{ std(data['session_utilization_cpus_percent_avg']) }} %**,
* *Median*: **{{ median(data['session_utilization_cpus_percent_avg']) }} %**.
{% endif %}

{% if 'session_utilization_mem_percent' in data -%}
## Memory usage

```{figure} {{data["memusagepath"]}}
---
name: {{basename}}_memusage
alt: Memory usage
align: center
---

Memory usage during benchmark
```

* *Mean*: **{{ mean(data['session_utilization_mem_percent']) }} %**,
* *Standard deviation*: **{{ std(data['session_utilization_mem_percent']) }} %**,
* *Median*: **{{ median(data['session_utilization_mem_percent']) }} %**.
{% endif %}

{% if 'session_utilization_gpu_utilization' in data and data['session_utilization_gpu_utilization']|length > 0 -%}
# GPU usage

```{figure} {{data["gpuusagepath"]}}
---
name: {{basename}}_gpuusage
alt: GPU usage
align: center
---

GPU utilization during benchmark
```

* *Mean*: **{{ mean(data['session_utilization_gpu_utilization']) }} %**,
* *Standard deviation*: **{{ std(data['session_utilization_gpu_utilization']) }} %**,
* *Median*: **{{ median(data['session_utilization_gpu_utilization']) }} %**.
{% endif %}

{% if 'session_utilization_gpu_mem_utilization' in data and data['session_utilization_gpu_mem_utilization']|length > 0 -%}
# GPU memory usage

```{figure} {{data["gpumemusagepath"]}}
---
name: {{basename}}_gpumemusage
alt: GPU memory usage
align: center
---

GPU memory usage during benchmark
```

* *Mean*: **{{ mean(data['session_utilization_gpu_mem_utilization']) }} MB**,
* *Standard deviation*: **{{ std(data['session_utilization_gpu_mem_utilization']) }} MB**,
* *Median*: **{{ median(data['session_utilization_gpu_mem_utilization']) }} MB**.
{% endif %}

