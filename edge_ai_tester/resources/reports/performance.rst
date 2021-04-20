Inference performance metrics
-----------------------------

General information
~~~~~~~~~~~~~~~~~~~

* *Model framework*: {{ data['model_framework'] }} ver. {{ data['model_version'] }}
{% if 'compiler_framework' in data -%}
* *Compiler framework*: {{ data['compiler_framework'] }} ver. {{ data['compiler_version'] }}
{% endif %}

{% if 'inferencetime' in data -%}
Inference time
~~~~~~~~~~~~~~

.. figure:: {{data["inferencetimepath"]}}
   :name: {{data["reportname"][0]}}_inferencetime
   :alt: Inference time
   :align: center

    Inference time

* *First inference duration* (usually including allocation time): **{{ data['inferencetime'][0] }}**,
* *Mean*: **{{ mean(data['inferencetime']) }} s**,
* *Standard deviation*: **{{ std(data['inferencetime']) }} s**,
* *Median*: **{{ median(data['inferencetime']) }} s**.
{% endif %}

{% if 'session_utilization_mem_percent' in data -%}
Memory usage
~~~~~~~~~~~~

.. figure:: {{data["memusagepath"]}}
   :name: {{data["reportname"][0]}}_memusage
   :alt: Memory usage
   :align: center

    Memory usage during benchmark

* *Mean*: **{{ mean(data['session_utilization_mem_percent']) }} %**,
* *Standard deviation*: **{{ std(data['session_utilization_mem_percent']) }} %**,
* *Median*: **{{ median(data['session_utilization_mem_percent']) }} %**.
{% endif %}

{% if 'session_utilization_gpu_mem_utilization' in data -%}
GPU memory usage
~~~~~~~~~~~~~~~~

.. figure:: {{data["gpumemusagepath"]}}
   :name: {{data["reportname"][0]}}_gpumemusage
   :alt: GPU memory usage
   :align: center

    GPU memory usage during benchmark

* *Mean*: **{{ mean(data['session_utilization_gpu_mem_utilization']) }} MB**,
* *Standard deviation*: **{{ std(data['session_utilization_gpu_mem_utilization']) }} MB**,
* *Median*: **{{ median(data['session_utilization_gpu_mem_utilization']) }} MB**.
{% endif %}

{% if 'session_utilization_gpu_percent' in data -%}
GPU usage
~~~~~~~~~

.. figure:: {{data["gpuusagepath"]}}
   :name: {{data["reportname"][0]}}_gpuusage
   :alt: GPU usage
   :align: center

    GPU utilization during benchmark

* *Mean*: **{{ mean(data['session_utilization_gpu_utilization']) }} %**,
* *Standard deviation*: **{{ std(data['session_utilization_gpu_utilization']) }} %**,
* *Median*: **{{ median(data['session_utilization_gpu_utilization']) }} %**.
{% endif %}

