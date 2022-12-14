## Inference comparison

### Performance metrics

{% if 'inference_step_path' in data -%}
```{figure} {{data["inference_step_path"]}}
---
name: {{data["reportname_simple"]}}_inference_step_comparison
alt: Inference time comparison
align: center
---

Plot represents changes of inference time over time for all models.
```
{% endif %}

{% if 'session_utilization_cpus_percent_path' in data -%}
```{figure} {{data["session_utilization_cpus_percent_path"]}}
---
name: {{data["reportname_simple"]}}_cpu_comparison
alt: CPU usage comparison
align: center
---

Plot represents changes of CPU usage over time for all models.
```
{% endif %}

{% if 'session_utilization_mem_percent_path' in data -%}
```{figure} {{data["session_utilization_mem_percent_path"]}}
---
name: {{data["reportname_simple"]}}_memory_comparison
alt: Memory usage comparison
align: center
---

Plot represents changes of RAM usage over time for all models.
```
{% endif %}

{% if 'session_utilization_gpu_utilization_path' in data -%}
```{figure} {{data["session_utilization_gpu_utilization_path"]}}
---
name: {{data["reportname_simple"]}}_gpu_comparison
alt: GPU usage comparison
align: center
---

Plot represents changes of GPU usage over time for all models.
```
{% endif %}

{% if 'session_utilization_gpu_mem_utilization_path' in data -%}
```{figure} {{data["session_utilization_gpu_mem_utilization_path"]}}
---
name: {{data["reportname_simple"]}}_gpu_mem_comparison
alt: GPU memory usage comparison
align: center
---

Plot represents changes of GPU RAM usage over time for all models.
```
{% endif %}

### Mean comparison plot

```{figure} {{data["meanperformancepath"]}}
---
name: {{data["reportname_simple"]}}_performance_comparison
alt: Performance comparison
align: center
---
Violin chart representing distribution of values for performance metrics for models
```

### Hardware usage comparison

```{figure} {{data["hardwareusagepath"]}}
---
name: {{data["reportname_simple"]}}_hardware_usage_comparison
alt: Resource usage comparison
align: center
---
Radar chart representing the resource usage of models
```


