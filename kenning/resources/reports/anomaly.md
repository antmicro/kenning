{%- if 'anomalyfarfdrplot' in data %}

## Detection quality metrics {% if data["model_name"] %} for {{data["model_name"]}}{% endif %}

```{figure} {{data["anomalyfarfdrplot"]}}
---
name: anomaly_far_fdr_plot
alt: Anomaly Fault Detection Rate and False Alarm Rate plot
align: center
---
Anomaly Fault Detection Rate and False Alarm Rate plot
```

```{figure} {{data["anomalyaddsplot"]}}
---
name: anomaly_add_plot
alt: Anomaly Average Detection Delay plot
align: center
---
Anomaly Average Detection Delay plot
```

{%- endif %}

{%- if 'anomaly_timeseries_metrics' in data %}

## Anomaly detection metrics:

```{list-table} Anomaly detection metrics:
---
header-rows: 1
align: center
---
* - Statistic
  - Value
{%- for metric,value in data['anomaly_timeseries_metrics'].items() %}
* - *{{ metric }}*
  - **{{ '%.6f' % value }}**
{%- endfor %}
```

{%- endif %}
