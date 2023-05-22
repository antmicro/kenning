## Renode performance measurements {% if data["modelname"] %} for {{data["modelname"]}}{% endif %}

{% set basename = data["reportname_simple"] if "modelname" not in data else data["reportname_simple"] + data["modelname"] %}

{%- if 'instrbarpath' in data %}
### Count of instructions used during inference

```{figure} {{data["instrbarpath"]}}
---
name: {{basename}}_instrbarplot
alt: Count of used instructions during inference
align: center
---

Count of used instructions during inference
```

{%- if 'vectorinstrbarpath' in data %}
```{figure} {{data["vectorinstrbarpath"]}}
---
name: {{basename}}_vectorinstrbarplot
alt: Utilization of V-Extension instructions during inference
align: center
---

Utilization of V-Extension instructions during inference
```
{%- endif %}
{%- endif %}

{%- if 'executedinstrplotpath' in data %}
### Executed instructions counters

{%- for cpu, plotpath in data['executedinstrplotpath'].items() %}
```{figure} {{plotpath}}
---
name: {{basename}}_{{cpu}}_executedinstrplotpath
alt: Figure showing count of executed instructions for {{cpu}}
align: center
---

Figure showing count of executed instructions for {{cpu}}
```
{%- endfor %}
{%- endif %}

{%- if 'memoryaccessesplotpath' in data %}
### Memory access counters

{%- if 'reads' in data['memoryaccessesplotpath'] %}
```{figure} {{data['memoryaccessesplotpath']['reads']}}
---
name: {{basename}}_memoryreadsplotpath
alt: Figure showing count of memory reads
align: center
---

Figure showing count of memory reads
```
{%- endif %}

{%- if 'writes' in data['memoryaccessesplotpath'] %}
```{figure} {{data['memoryaccessesplotpath']['writes']}}
---
name: {{basename}}_memorywritessplotpath
alt: Figure showing count of memory writes
align: center
---

Figure showing count of memory writes
```
{%- endif %}
{%- endif %}

{%- if 'peripheralaccessesplotpath' in data %}
### Peripheral access counters

{%- for peripheral, plotpath in data['peripheralaccessesplotpath'].items() %}
```{figure} {{plotpath['reads']}}
---
name: {{basename}}_{{peripheral}}_peripheralreadsplotpath
alt: Figure showing count of {{peripheral}} reads
align: center
---

Figure showing count of {{peripheral}} reads
```

```{figure} {{plotpath['writes']}}
---
name: {{basename}}_{{peripheral}}_peripheralwritesplotpath
alt: Figure showing count of {{peripheral}} writes
align: center
---

Figure showing count of {{peripheral}} writes
```
{%- endfor %}
{%- endif %}

{%- if 'exceptionsplotpath' in data %}
### Exceptions counters

```{figure} {{data['exceptionsplotpath']}}
---
name: {{basename}}_exceptionsplotpath
alt: Figure showing count of raised exceptions
align: center
---

Figure showing count of raised exceptions
```
{%- endif %}

### Memory allocation stats

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

