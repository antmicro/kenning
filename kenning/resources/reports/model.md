## Original model specification{% if data["model_name"] %} for {{data["model_name"]}}{% endif %}

### Basic model information

```{list-table} Basic model information
---
header-rows: 1
align: center
---
* - Statistic
  - Value

* - Layer Count
  - **{{data["layer count"]}}**
* - Number of parameters
  - **{{data["total parameters"]}}**
* - Size (in bytes)
  - **{{data["total bytes"]}}**
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
