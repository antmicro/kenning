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

```{list-table} Inference time metrics
---
header-rows: 1
align: center
---

* - Statistic
  - Time [s]

* - First inference duration
  - **{{'%.6f' % data['inferencetime'][0] }}**
* - Mean
  - **{{'%.6f' % data['inferencetime_mean'] }}**
* - Median
  - **{{'%.6f' % data['inferencetime_median'] }}**
* - Standard deviation
  - **{{'%.6f' % data['inferencetime_std'] }}**
* - Minimum
  - **{{'%.6f' % data['inferencetime_min'] }}**
* - Maximum
  - **{{'%.6f' % data['inferencetime_max'] }}**
```
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

```{list-table} CPU usage metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Usage [%]

* - Mean
  - **{{'%.3f' %  data['session_utilization_cpus_percent_avg_mean'] }}**
* - Standard deviation
  - **{{'%.3f' %  data['session_utilization_cpus_percent_avg_std'] }}**
* - Median
  - **{{'%.3f' %  data['session_utilization_cpus_percent_avg_median'] }}**

```
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

```{list-table} Memory usage metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Usage [%]

* - Mean
  - **{{'%.3f' %  data['session_utilization_mem_percent_mean'] }}**
* - Standard deviation
  - **{{'%.3f' %  data['session_utilization_mem_percent_std'] }}**
* - Median
  - **{{'%.3f' %  data['session_utilization_mem_percent_median'] }}**

```
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

```{list-table} GPU utilization metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Usage [%]

* - Mean
  - **{{'%.3f' % data['session_utilization_gpu_utilization_mean'] }}**
* - Standard deviation
  - **{{'%.3f' % data['session_utilization_gpu_utilization_std'] }}**
* - Median
  - **{{'%.3f' % data['session_utilization_gpu_utilization_median'] }}**

```
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

```{list-table} GPU memory usage metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Memory used [MB]

* - Mean
  - **{{'%.6f' % data['session_utilization_gpu_mem_utilization_mean'] }}**
* - Standard deviation
  - **{{'%.6f' % data['session_utilization_gpu_mem_utilization_std'] }}**
* - Median
  - **{{'%.6f' % data['session_utilization_gpu_mem_utilization_median'] }}**

```
{% endif %}

