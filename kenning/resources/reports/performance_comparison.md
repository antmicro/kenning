# Inference comparison

## Performance metrics

{% if 'inference_step_path' in data -%}
```{figure} {{data["inference_step_path"]}}
---
name: {{data["reportname"]}}_inference_step_comparison
alt: Inference step comparison
align: center
---
```
{% endif %}

{% if 'session_utilization_cpus_percent_path' in data -%}
```{figure} {{data["session_utilization_cpus_percent_path"]}}
---
name: {{data["reportname"]}}_cpu_comparison
alt: CPU usage comparison
align: center
---
```
{% endif %}

{% if 'session_utilization_mem_percent_path' in data -%}
```{figure} {{data["session_utilization_mem_percent_path"]}}
---
name: {{data["reportname"]}}_memory_comparison
alt: Memory usage comparison
align: center
---
```
{% endif %}

{% if 'session_utilization_gpu_utilization_path' in data -%}
```{figure} {{data["session_utilization_gpu_utilization_path"]}}
---
name: {{data["reportname"]}}_gpu_comparison
alt: GPU usage comparison
align: center
---
```
{% endif %}

{% if 'session_utilization_gpu_mem_utilization_path' in data -%}
```{figure} {{data["session_utilization_gpu_mem_utilization_path"]}}
---
name: {{data["reportname"]}}_gpu_mem_comparison
alt: GPU memory usage comparison
align: center
---
```
{% endif %}

## Mean comparison plot

```{figure} {{data["meanperformancepath"]}}
---
name: {{data["reportname"]}}_performance_comparison
alt: Performance comparison
align: center
---
```

## Hardware usage comparison

```{figure} {{data["hardwareusagepath"]}}
---
name: {{data["reportname"]}}_hardware_usage_comparison
alt: Resource usage comparison
align: center
---
```


