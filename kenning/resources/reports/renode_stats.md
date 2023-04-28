## Inference Renode metrics{% if data["modelname"] %} for {{data["modelname"]}}{% endif %}

{% set basename = data["reportname_simple"] if "modelname" not in data else data["reportname_simple"] + data["modelname"] %}
```{figure} {{data["instrhistpath"]}}
---
name: {{basename}}_instrhistogram
alt: Opcode histogram
align: center
---

Opcode histogram
```

```{figure} {{data["vectorinstrhistpath"]}}
---
name: {{basename}}_vectorinstrhistogram
alt: Vector opcode histogram
align: center
---

Vector instructions histogram
```

{%- if data['host_bytes_peak'] %}
* *Host bytes peak*: **{{ data['host_bytes_peak'] }}**
{%- endif %}
{%- if data['host_bytes_allocated'] %}
* *Host bytes allocated*: **{{ data['host_bytes_allocated'] }}**
{%- endif %}
{%- if data['host_bytes_freed'] %}
* *Host bytes freed*: **{{ data['host_bytes_freed'] }}**
{%- endif %}
{%- if data['device_bytes_peak'] %}
* *Device bytes peak*: **{{ data['device_bytes_peak'] }}**
{%- endif %}
{%- if data['device_bytes_allocated'] %}
* *Device bytes allocated*: **{{ data['device_bytes_allocated'] }}**
{%- endif %}
{%- if data['device_bytes_freed'] %}
* *Device bytes freed*: **{{ data['device_bytes_freed'] }}**
{%- endif %}
{%- if data['compiled_model_size'] %}
* *Compiled model size*: **{{ data['compiled_model_size'] }}**
{%- endif %}

