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

{% if 'pre_post_process_path' in data -%}
```{figure} {{data["pre_post_process_path"]}}
---
name: {{basename}}_pre_post_process
alt: Preprocessing and postprocessing time
align: center
---

Preprocessing and postprocessing time
```
{% endif %}

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

{% if 'cpu_system_usage_path' in data -%}
### Average CPU usage by system

```{figure} {{data["cpu_system_usage_path"]}}
---
name: {{basename}}_cpuusage
alt: Average CPU usage by system
align: center
---

Average CPU usage by system during benchmark
```

```{list-table} CPU usage by system metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Usage [%]

* - Mean
  - **{{'%.3f' %  data['session_utilization_system_cpu_percent_mean'] }}**
* - Standard deviation
  - **{{'%.3f' %  data['session_utilization_system_cpu_percent_std'] }}**
* - Median
  - **{{'%.3f' %  data['session_utilization_system_cpu_percent_median'] }}**

```
{% endif %}

{% if 'cpu_process_usage_path' in data -%}
### Average CPU usage by process

```{figure} {{data["cpu_process_usage_path"]}}
---
name: {{basename}}_cpuusage
alt: Average CPU usage by process
align: center
---

Average CPU usage by process during benchmark
```
{% endif %}

{% if 'cpu_all_usage_path' in data -%}
### Average CPU usage by system and process

```{figure} {{data["cpu_all_usage_path"]}}
---
name: {{basename}}_cpuusage
alt: Average CPU usage by system and process
align: center
---

Average CPU usage by system and processes during benchmark
```

```{list-table} CPU usage by system metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Usage [%]

* - Mean
  - **{{'%.3f' %  data['session_utilization_system_cpu_percent_mean'] }}**
* - Standard deviation
  - **{{'%.3f' %  data['session_utilization_system_cpu_percent_std'] }}**
* - Median
  - **{{'%.3f' %  data['session_utilization_system_cpu_percent_median'] }}**

```

{% endif %}

{% if 'cpu_mem_system_usage_path' in data -%}
### RAM usage by system

```{figure} {{data["cpu_mem_system_usage_path"]}}
---
name: {{basename}}_system_memusage
alt: RAM usage by system
align: center
---

RAM usage by system during benchmark
```

```{list-table} RAM usage by system metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Usage [%]

* - Mean
  - **{{'%.3f' %  data['session_utilization_system_cpu_mem_percent_mean'] }}**
* - Standard deviation
  - **{{'%.3f' %  data['session_utilization_system_cpu_mem_percent_std'] }}**
* - Median
  - **{{'%.3f' %  data['session_utilization_system_cpu_mem_percent_median'] }}**

```
{% endif %}

{% if 'cpu_mem_process_usage_path' in data -%}
### RAM usage by process

```{figure} {{data["cpu_mem_process_usage_path"]}}
---
name: {{basename}}_process_memusage
alt: RAM usage by process
align: center
---

RAM usage by process during benchmark
```

```{list-table} RAM usage by process metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Usage [%]

* - Mean
  - **{{'%.3f' %  data['session_utilization_process_cpu_mem_percent_mean'] }}**
* - Standard deviation
  - **{{'%.3f' %  data['session_utilization_process_cpu_mem_percent_std'] }}**
* - Median
  - **{{'%.3f' %  data['session_utilization_process_cpu_mem_percent_median'] }}**

```
{% endif %}

{% if 'cpu_mem_all_usage_path' in data -%}
### RAM usage by system and process

```{figure} {{data["cpu_mem_all_usage_path"]}}
---
name: {{basename}}_all_memusage
alt: RAM usage by system and process
align: center
---

RAM usage by system and process during benchmark
```

```{list-table} RAM usage by system metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Usage [%]

* - Mean
  - **{{'%.3f' %  data['session_utilization_system_cpu_mem_percent_mean'] }}**
* - Standard deviation
  - **{{'%.3f' %  data['session_utilization_system_cpu_mem_percent_std'] }}**
* - Median
  - **{{'%.3f' %  data['session_utilization_system_cpu_mem_percent_median'] }}**

```

```{list-table} RAM usage by process metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Usage [%]

* - Mean
  - **{{'%.3f' %  data['session_utilization_process_cpu_mem_percent_mean'] }}**
* - Standard deviation
  - **{{'%.3f' %  data['session_utilization_process_cpu_mem_percent_std'] }}**
* - Median
  - **{{'%.3f' %  data['session_utilization_process_cpu_mem_percent_median'] }}**

```
{% endif %}

{% if 'gpu_system_usage_path' in data -%}
### GPU usage by system


```{figure} {{data["gpu_system_usage_path"]}}
---
name: {{basename}}_gpuusage
alt: GPU usage by system
align: center
---

GPU utilization by system during benchmark
```

```{list-table} GPU utilization by system metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Usage [%]

* - Mean
  - **{{'%.3f' % data['session_utilization_system_gpu_percent_mean'] }}**
* - Standard deviation
  - **{{'%.3f' % data['session_utilization_system_gpu_percent_std'] }}**
* - Median
  - **{{'%.3f' % data['session_utilization_system_gpu_percent_median'] }}**

```
{% endif %}

{% if 'gpu_process_usage_path' in data -%}
### GPU usage by process


```{figure} {{data["gpu_process_usage_path"]}}
---
name: {{basename}}_gpuusage
alt: GPU usage by process
align: center
---

GPU utilization by process during benchmark
```

```{list-table} GPU utilization by process metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Usage [%]

* - Mean
  - **{{'%.3f' % data['session_utilization_process_gpu_percent_mean'] }}**
* - Standard deviation
  - **{{'%.3f' % data['session_utilization_process_gpu_percent_std'] }}**
* - Median
  - **{{'%.3f' % data['session_utilization_process_gpu_percent_median'] }}**

```
{% endif %}

{% if 'gpu_all_usage_path' in data -%}
### GPU usage by system and process

```{figure} {{data["gpu_all_usage_path"]}}
---
name: {{basename}}_gpuusage
alt: GPU usage
align: center
---

GPU utilization by system and process during benchmark
```

```{list-table} GPU utilization by system metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Usage [%]

* - Mean
  - **{{'%.3f' % data['session_utilization_system_gpu_percent_mean'] }}**
* - Standard deviation
  - **{{'%.3f' % data['session_utilization_system_gpu_percent_std'] }}**
* - Median
  - **{{'%.3f' % data['session_utilization_system_gpu_percent_median'] }}**

```

{% if 'session_utilization_process_gpu_percent_mean' in data -%}
```{list-table} GPU utilization by process metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Usage [%]

* - Mean
  - **{{'%.3f' % data['session_utilization_process_gpu_percent_mean'] }}**
* - Standard deviation
  - **{{'%.3f' % data['session_utilization_process_gpu_percent_std'] }}**
* - Median
  - **{{'%.3f' % data['session_utilization_process_gpu_percent_median'] }}**

```
{% endif %}
{% endif %}

{% if 'gpu_mem_system_usage_path' in data -%}
### GPU memory usage by system

```{figure} {{data["gpu_mem_system_usage_path"]}}
---
name: {{basename}}_gpumemusage
alt: GPU memory usage by system
align: center
---

GPU memory usage by system during benchmark
```

```{list-table} GPU memory usage by system metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Memory used [%]

* - Mean
  - **{{'%.6f' % data['session_utilization_system_gpu_mem_percent_mean'] }}**
* - Standard deviation
  - **{{'%.6f' % data['session_utilization_system_gpu_mem_percent_std'] }}**
* - Median
  - **{{'%.6f' % data['session_utilization_system_gpu_mem_percent_median'] }}**

```
{% endif %}

{% if 'gpu_mem_process_usage_path' in data -%}
### GPU memory usage by process

```{figure} {{data["gpu_mem_process_usage_path"]}}
---
name: {{basename}}_gpumemusage
alt: GPU memory usage by process
align: center
---

GPU memory usage by process during benchmark
```

```{list-table} GPU memory usage by process metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Memory used [%]

* - Mean
  - **{{'%.6f' % data['session_utilization_process_gpu_mem_percent_mean'] }}**
* - Standard deviation
  - **{{'%.6f' % data['session_utilization_process_gpu_mem_percent_std'] }}**
* - Median
  - **{{'%.6f' % data['session_utilization_process_gpu_mem_percent_median'] }}**

```
{% endif %}

{% if 'gpu_mem_all_usage_path' in data -%}
### GPU memory usage by system and process

```{figure} {{data["gpu_mem_all_usage_path"]}}
---
name: {{basename}}_gpumemusage
alt: GPU memory usage by system and process
align: center
---

GPU memory usage by system and process during benchmark
```

```{list-table} GPU memory usage by system metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Memory used [%]

* - Mean
  - **{{'%.6f' % data['session_utilization_system_gpu_mem_percent_mean'] }}**
* - Standard deviation
  - **{{'%.6f' % data['session_utilization_system_gpu_mem_percent_std'] }}**
* - Median
  - **{{'%.6f' % data['session_utilization_system_gpu_mem_percent_median'] }}**

```

{% if 'session_utilization_process_gpu_mem_percent_mean' in data -%}
```{list-table} GPU memory usage by process metrics
---
header-rows: 1
align: center
---
* - Statistic
  - Memory used [%]

* - Mean
  - **{{'%.6f' % data['session_utilization_process_gpu_mem_percent_mean'] }}**
* - Standard deviation
  - **{{'%.6f' % data['session_utilization_process_gpu_mem_percent_std'] }}**
* - Median
  - **{{'%.6f' % data['session_utilization_process_gpu_mem_percent_median'] }}**

```
{% endif %}
{% endif %}
