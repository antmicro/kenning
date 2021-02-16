{% set title = 'Classification performance for ' ~ data["framework"][0] ~ ' ver. ' ~ data["version"][0] %}

{{ title }}
{{ '=' * title|length }}

.. figure:: {{data["memusagepath"][0]}}
   :name: {{data["reportname"][0]}}_memoryusage
   :alt: Memory usage for {{data["reportname"][0]}}
   :align: center

    Memory usage

.. figure:: {{data["batchtimepath"][0]}}
   :name: {{data["reportname"][0]}}_batchtime
   :alt: Memory usage for {{data["reportname"][0]}}
   :align: center

    Inference time per batch over time.

