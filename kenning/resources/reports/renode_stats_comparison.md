## Renode performance measurements {% if data["model_name"] %} for {{data["model_name"]}}{% endif %}

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
```{figure} {{plotpath['persec']}}
---
name: {{basename}}_{{cpu}}_executedinstrplotpath
alt: Count of executed instructions per second for {{cpu}}
align: center
---

Count of executed instructions per second for {{cpu}} during benchmark
```

```{figure} {{plotpath['cumulative']}}
---
name: {{basename}}_{{cpu}}_executedinstrplotpath
alt: Cumulative count of executed instructions for {{cpu}}
align: center
---

Cumulative count of executed instructions for {{cpu}} during benchmark
```
{%- endfor %}
{%- endif %}

{%- if 'memoryaccessesplotpath' in data %}
### Memory access counters
{%- if 'read' in data['memoryaccessesplotpath'] %}
```{figure} {{data['memoryaccessesplotpath']['read']['persec']}}
---
name: {{basename}}_memoryreadsplotpath
alt: Count of memory reads per second
align: center
---

Count of memory reads per second during benchmark
```

```{figure} {{data['memoryaccessesplotpath']['read']['cumulative']}}
---
name: {{basename}}_memoryreadsplotpath
alt: Cumulative count of memory reads
align: center
---

Cumulative count of memory reads during benchmark
```
{%- endif %}

{%- if 'write' in data['memoryaccessesplotpath'] %}
```{figure} {{data['memoryaccessesplotpath']['write']['persec']}}
---
name: {{basename}}_memorywritessplotpath
alt: Count of memory writes per second
align: center
---

Count of memory writes per second during benchmark
```

```{figure} {{data['memoryaccessesplotpath']['write']['cumulative']}}
---
name: {{basename}}_memorywritessplotpath
alt: Cumulative count of memory writes
align: center
---

Cumulative count of memory writes during benchmark
```
{%- endif %}
{%- endif %}

{%- if 'peripheralaccessesplotpath' in data %}
### Peripheral access counters

{%- for peripheral, plotpath in data['peripheralaccessesplotpath'].items() %}
```{figure} {{plotpath['read']['persec']}}
---
name: {{basename}}_{{peripheral}}_peripheralreadsplotpath
alt: Count of {{peripheral}} reads per second
align: center
---

Count of {{peripheral}} reads per second during benchmark
```

```{figure} {{plotpath['read']['cumulative']}}
---
name: {{basename}}_{{peripheral}}_peripheralreadsplotpath
alt: Cumulative count of {{peripheral}} reads
align: center
---

Cumulative count of {{peripheral}} reads during benchmark
```

```{figure} {{plotpath['write']['persec']}}
---
name: {{basename}}_{{peripheral}}_peripheralwritesplotpath
alt: Count of {{peripheral}} writes per second
align: center
---

Count of {{peripheral}} writes per second during benchmark
```

```{figure} {{plotpath['write']['cumulative']}}
---
name: {{basename}}_{{peripheral}}_peripheralwritesplotpath
alt: Cumulative count of {{peripheral}} writes
align: center
---

Cumulative count of {{peripheral}} writes during benchmark
```
{%- endfor %}
{%- endif %}

{%- if 'exceptionsplotpath' in data %}
### Exceptions counters

```{figure} {{data['exceptionsplotpath']['persec']}}
---
name: {{basename}}_exceptionsplotpath
alt: Count of raised exceptions per second
align: center
---

Count of raised exceptions per second during benchmark
```

```{figure} {{data['exceptionsplotpath']['cumulative']}}
---
name: {{basename}}_exceptionsplotpath
alt: Cumulative count of raised exceptions
align: center
---

Cumulative count of raised exceptions during benchmark
```
{%- endif %}

