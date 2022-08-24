Inference performance metrics
-----------------------------

{% if data['command']|length > 0 -%}
.. note::

    This section was generated using:

    .. code-block:: bash

        {% for line in data['command'] -%}
        {{ line }}
        {% endfor %}
{% endif -%}

{% if 'build_cfg' in data -%}
.. note::
    Input JSON:

    .. code-block:: json

        {% for line in data['build_cfg'] -%}
        {{ line }}
        {% endfor %}
{% endif -%}

General information
~~~~~~~~~~~~~~~~~~~

*Model framework*:

* {{ data['model_framework'] }} ver. {{ data['model_version'] }}
{% if data['compilers']|length > 0 %}
    {% if data['compilers']|length == 1 %}
*Compiler framework*:
    {% else %}
*Compiler frameworks*:
    {%- endif %}
    {% for line in data['compilers'] %}
* {{ line['compiler_framework'] }} ver. {{ line['compiler_version'] }}
    {%- endfor %}
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

{% if 'session_utilization_cpus_percent_avg' in data -%}
Mean CPU usage
~~~~~~~~~~~~~~

.. figure:: {{data["cpuusagepath"]}}
    :name: {{data["reportname"][0]}}_cpuusage
    :alt: Mean CPU usage
    :align: center

    Mean CPU usage during benchmark

* *Mean*: **{{ mean(data['session_utilization_cpus_percent_avg']) }} %**,
* *Standard deviation*: **{{ std(data['session_utilization_cpus_percent_avg']) }} %**,
* *Median*: **{{ median(data['session_utilization_cpus_percent_avg']) }} %**.
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

{% if 'session_utilization_gpu_utilization' in data and data['session_utilization_gpu_utilization']|length > 0 -%}
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

{% if 'session_utilization_gpu_mem_utilization' in data and data['session_utilization_gpu_mem_utilization']|length > 0 -%}
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

