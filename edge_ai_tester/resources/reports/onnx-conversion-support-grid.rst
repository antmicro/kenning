ONNX conversion support grid
----------------------------

.. list-table:: ONNX conversion support grid
    :header-rows: 1
    :align: center
    :name: onnx-support-grid

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

The :numref:`onnx-support-grid` table shows the ONNX conversion support for some of the popular models across various deep learning frameworks.
Each row represents different deep learning model.
Each column represents different deep learning framework.
Each cell shows export to ONNX and import from ONNX support for a given model and framework.

First of all, the model is downloaded for a given framework.
Secondly, the model is converted to ONNX.
In the end, the ONNX model is converted back to the framework's format.

The values in cells are in ``<export support> / <import support>`` format.

The possible values are:

* ``supported`` if export or import succeded,
* ``unsupported`` if export or import is not implemented for a given framework,
* ``ERROR`` if export or import ended up with error for a given framework,
* ``unverified`` if import could not be tested due to lack of support for export or error during export.
