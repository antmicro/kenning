## Inference Renode metrics{% if data["modelname"] %} for {{data["modelname"]}}{% endif %}

{% set basename = data["reportname_simple"] if "modelname" not in data else data["reportname_simple"] + data["modelname"] %}

### Executed instructions opcodes stats

```{figure} {{data["instrbarpath"]}}
---
name: {{basename}}_instrbarplot
alt: Opcodes barplot
align: center
---

Opcodes barplot
```

{%- if data['vectorinstrbarpath'] %}
```{figure} {{data["vectorinstrbarpath"]}}
---
name: {{basename}}_vectorinstrbarplot
alt: Vector opcodes barplot
align: center
---

Vector opcodes barplot
```
{%- endif %}

### Executed instructions counters

{%- for cpu, plotpath in data['executedinstrplotpath'].items() %}
```{figure} {{plotpath}}
---
name: {{basename}}_{{cpu}}_executedinstrplotpath
alt: Executed instructions plot for {{cpu}}
align: center
---

Executed instructions plot for {{cpu}}
```
{%- endfor %}

### Memory access counters

```{figure} {{data['memoryaccessesplotpath']['reads']}}
---
name: {{basename}}_memoryreadsplotpath
alt: Memory reads plot
align: center
---

Memory reads plot
```

```{figure} {{data['memoryaccessesplotpath']['writes']}}
---
name: {{basename}}_memorywritessplotpath
alt: Memory writes plot
align: center
---

Memory writes plot
```

### Peripheral access counters

{%- for peripheral, plotpath in data['peripheralaccessesplotpath'].items() %}
```{figure} {{plotpath['reads']}}
---
name: {{basename}}_{{peripheral}}_peripheralreadsplotpath
alt: Peripheral reads for {{peripheral}}
align: center
---

Peripheral reads for {{peripheral}}
```

```{figure} {{plotpath['writes']}}
---
name: {{basename}}_{{peripheral}}_peripheralwritesplotpath
alt: Peripheral writes for {{peripheral}}
align: center
---

Peripheral writes for {{peripheral}}
```
{%- endfor %}

{%- if data['exceptionsplotpath'] %}
### Exceptions counters

```{figure} {{data['exceptionsplotpath']}}
---
name: {{basename}}_exceptionsplotpath
alt: Exceptions plot
align: center
---

Exceptions plot
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

