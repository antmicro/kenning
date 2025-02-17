{% if data['smaller_header'] %}
## {{data['report_name']}}
{% else %}
# {{data['report_name']}}
{% endif %}

{% if data['command']|length > 0 -%}
### Commands used

````{note}

This section was generated using:

```bash
{% for line in data['command'] -%}
{{ line }}
{% endfor %}
```
````
{% endif -%}


{% for model_name in data['model_names'] %}
{% if not data[model_name]["__unoptimized__"] is defined %}
### General information for {{model_name}}

*Model framework*:

* {{ data[model_name]['model_framework'] }} ver. {{ data[model_name]['model_version'] }}
{% if data[model_name]['compilers']|length > 0 %}
    {% if data[model_name]['compilers']|length == 1 %}
*Compiler framework*:
    {% else %}
*Compiler frameworks*:
    {%- endif %}
    {% for line in data[model_name]['compilers'] %}
* {{ line['compiler_framework'] }} ver. {{ line['compiler_version'] }}
    {%- endfor %}
{% endif %}
{% if 'build_cfg' in data[model_name] -%}
*Input JSON*:

```json
{% for line in data[model_name]['build_cfg'] -%}
{{ line }}
{% endfor %}
```
{% endif -%}
{% endif -%}
{% endfor %}
