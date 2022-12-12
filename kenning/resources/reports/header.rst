{{data['reportname']}}
{{'-' * data['reportname']|length}}

Commands used
~~~~~~~~~~~~~

{% if data['command']|length > 0 -%}
.. note::

    This section was generated using:

    .. code-block:: bash

        {% for line in data['command'] -%}
        {{ line }}
        {% endfor %}
{% endif -%}


{% for modelname in data['modelnames'] %}
General information for {{modelname}}
~~~~~~~~~~~~~~~~~~~~~~~~{{'~' * modelname|length}}

*Model framework*:

* {{ data[modelname]['model_framework'] }} ver. {{ data[modelname]['model_version'] }}
{% if data[modelname]['compilers']|length > 0 %}
    {% if data[modelname]['compilers']|length == 1 %}
*Compiler framework*:
    {% else %}
*Compiler frameworks*:
    {%- endif %}
    {% for line in data[modelname]['compilers'] %}
* {{ line['compiler_framework'] }} ver. {{ line['compiler_version'] }}
    {%- endfor %}
{% endif %}
{% if 'build_cfg' in data[modelname] -%}
*Input JSON*:

.. note::

    .. code-block:: json

        {% for line in data[modelname]['build_cfg'] -%}
        {{ line }}
        {% endfor %}
{% endif -%}
{% endfor %}
