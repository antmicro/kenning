{% set title = 'Device performance metrics for ' ~ data["framework"][0] ~ ' ver. ' ~ data["version"][0] %}

{{ title }}
{{ '-' * title|length }}

.. figure:: {{data["memusagepath"]}}
   :name: {{data["reportname"]}}_memoryusage
   :alt: Memory usage for {{data["reportname"]}}
   :align: center

    Memory usage

.. figure:: {{data["inferencetimepath"]}}
   :name: {{data["reportname"]}}_batchtime
   :alt: Memory usage for {{data["reportname"]}}
   :align: center

    Inference time per batch over time.

