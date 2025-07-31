## Detection comparison

### mAP over thresholds comparison

```{figure} {{data["mapcomparisonpath"]}}
---
name: {{data["report_name_simple"]}}_map_comparison
alt: mAP comparison over thresholds
align: center
---
The plot demonstrates the change of mAP metric depending on objectness threshold for all compared models
```

```{list-table} Best Mean Average Precision of each model and corresponding thresholds
---
header-rows: 1
align: center
---
* - Model name
  - Threshold
  - mAP value
{%- for model_name in data["model_names"] %}
* - {{model_name}}
  - {{data[model_name]["best_map_thr"]}}
  - {{data[model_name]["best_map"]}}
{%- endfor %}
```


