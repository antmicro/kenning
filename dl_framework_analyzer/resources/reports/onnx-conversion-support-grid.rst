ONNX conversion support grid
============================

This section shows the support for importing and exporting models from various frameworks to ONNX.

The support is shown in the:

.. list-table:: ONNX conversion support grid
    :header-rows: 1
    :align: center

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
