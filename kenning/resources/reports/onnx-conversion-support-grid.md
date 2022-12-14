## ONNX conversion support grid

{% if command|length > 0 -%}
``````{note}

This section was generated using:

```bash

{% for line in command -%}
{{ line }}
{% endfor %}
```
``````
{% endif -%}


```{list-table} ONNX conversion support grid
---
header-rows: 1
align: center
name: onnx-support-grid
---

* - Model Name
  {% for framework in headers -%}
  - {{ framework }}
  {% endfor %}
{% for key, value in grid.items() -%}
* - {{ key }}
  {% for framework in headers -%}
  - {{ value[framework] }}
  {% endfor %}
{% endfor %}
```

{figure:numref}`onnx-support-grid` table shows ONNX conversion support for several of the popular models across various deep learning frameworks.
Each row represents a different deep learning model.
Each column represents a different deep learning framework.
Each cell lists an export to ONNX and import from ONNX support status for a given model and framework.

Firstly, the model is downloaded for a given framework.
Secondly, the model is converted to ONNX.
Lastly, the ONNX model is converted back to the framework's format.

The values in cells are in `<export support> / <import support>` format.

Possible values are:

* `supported` if export or import was successful,
* `unsupported` if export or import is not implemented for a given framework,
* `ERROR` if export or import ended up with error for a given framework,
* `unverified` if import could not be tested due to lack of support for export or an error during export.
* `Not provided` if the model was not provided for the framework.
