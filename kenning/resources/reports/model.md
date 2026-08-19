## Input model specification{% if data["model_name"] %} for {{data["model_name"]}}{% endif %}

### Basic model information

```{list-table} Basic model information
---
header-rows: 1
align: center
---
* - Model name
  - Layer count
  - Number of parameters
  - ONNX-converted model size (in bytes)
  - Model Framework (version)
{% for result in data["layer_tables"] %}
* - {{ result["model_name"] }}
  - **{{ result["total"]["layer_count"] }}**
  - **{{ result["total"]["parameters"] }}**
  - **{{ result["total"]["bytes"] }}**
  - **{{ result["model_type"] }} ({{ data["framework_version"] }})**
{%- endfor %}
```

### Layers

{%- if 'layer_type_count_bar_path' in data %}
```{figure} {{data["layer_type_count_bar_path"]}}
---
name: layer_type_count_bar_path
alt: Layer Operations
align: center
---

Layer operation type counts
```
{% endif %}

```{table}
{% for result in data["layer_tables"] %}
```{table} {{ result["model_name"] }} model layers
---
align: center
---

|{% for param in result["columns"] %} {{param}} |{% endfor %}
|{% for param in result["columns"] %} :--- |{% endfor %}
{%- for layer in result["rows"] %}
|{% for param in layer %} {{param}} |{% endfor %}
{%- endfor %}

```
{% endfor %}
