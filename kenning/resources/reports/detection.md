## Object detection metrics{% if data["model_name"] %} for {{data["model_name"]}}{% endif %}

{% set basename = data["report_name_simple"] if "model_name" not in data else data["report_name_simple"] + data["model_name"] %}
```{figure} {{data["curvepath"]}}
---
name: {{basename}}_recall_precision_curves
alt: Recall-Precision curves
align: center
---

Per-Class Recall-Precision curves
```

```{figure} {{data["gradientpath"]}}
---
name: {{basename}}_recall_precision_gradients
alt: Per-Class precision gradients
align: center
---

Per-Class precision gradients
```

```{figure} {{data["mappath"]}}
---
name: {{basename}}_map
alt: mAP values depending on threshold
align: center
---

mAP values depending on threshold
```

{% if "map_best_recordings" in data %}
```{figure} {{data["map_best_recordings"]}}
---
name: {{basename}}_map_best_recordings
alt: mAP values depending on threshold for the most detected recordings
align: center
---

mAP values depending on threshold for the most detected recordings
```
{% endif %}

{% if "map_worst_recordings" in data %}
```{figure} {{data["map_worst_recordings"]}}
---
name: {{basename}}_map_worst_recordings
alt: mAP values depending on threshold for the least detected recordings
align: center
---

mAP values depending on threshold for the least detected recordings
```
{% endif %}

* *Mean Average Precision* for threshold 0.5: {{data['mAP']}}
* Best *Mean Average Precision* occurs at threshold {{data['max_mAP_index']}}  and it is: {{data['max_mAP']}}

```{figure} {{data["tpioupath"]}}
---
name: {{basename}}_tpiou
alt: Per-Class mean IoU values for correctly labeled objects
align: center
---

Per-Class mean IoU values for correctly labeled objects
```

```{figure} {{data["iouhistpath"]}}
---
name: {{basename}}_iouhist
alt: Histogram of IoU values for correctly labeled objects
align: center
---

Histogram of IoU values for correctly labeled objects
```


