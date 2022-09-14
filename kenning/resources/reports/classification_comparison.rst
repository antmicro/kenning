Classification comparison
-------------------------

.. figure:: {{data["bubbleplotpath"]}}
    :name: {{data["reportname"][0]}}_bubble
    :alt: Accuracy vs Inference time vs Ram usage
    :align: center

Metric comparison
~~~~~~~~~~~~~~~~~

.. figure:: {{data['radarchartpath']}}
    :name: {{data['reportname'][0]}}_radar
    :alt: Metric comparison
    :align: center

.. list-table::
    :header-rows: 1

    * - Model name
      - Accuracy
      - Mean precison
      - Mean recall
{% for modelname in data["modelnames"] %}
    * - {{modelname}}
      - {{data[modelname][0]}}
      - {{data[modelname][1]}}
      - {{data[modelname][2]}}
{% endfor %}

