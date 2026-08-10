## Input model specification{% if data["model_name"] %} for {{data["model_name"]}}{% endif %}

### Basic model information

```{list-table} Basic model information
---
header-rows: 1
align: center
---
* - Information
  - Value

* - Layer Count
  - **{{data["layer count"]}}**
* - Number of parameters
  - **{{data["total parameters"]}}**
* - ONNX-converted model size (in bytes)
  - **{{data["total bytes"]}}**
* - Input model framework (version)
  - **{{data["base model type"]}} ({{data["framework version"]}})**
* - Input model size (in bytes)
  - **{{data["base model size"]}}**
```

### Layers

```{table} Model's layers
---
align: center
---

|{% for param in data["layer statistics"] %} {{param}} |{% endfor %}
|{% for param in data["layer statistics"] %} :--- |{% endfor %}
{%- for layer in data["layers"] %}
|{% for param in layer %} {{param}} |{% endfor %}
{%- endfor %}

```
