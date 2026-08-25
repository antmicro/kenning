## CoralNPU compilation comparison

{% set basename = data["report_name_simple"] if "model_name" not in data else data["report_name_simple"] + data["model_name"] %}

{%- if "compilation_duration" in data %}
### Duration of compilation

```{list-table} Comparison of compilation times
---
header-rows: 1
align: center
---
* - Model
  - Compilation time [s]
{%- for name, t in data["compilation_duration"].items() %}
* - {{ name }}
  - {{ "{:.3f}".format(t) }}
{%- endfor %}
```
{%- endif %}

{%- if "dispatch_distribution_path" in data %}
### Dispatch distribution

```{figure} {{data["dispatch_distribution_path"]}}
---
name: {{basename}}_dispatch_distribution_path
alt: Comparison of dispatch distribution
align: center
---

Distribution of dispatches
```

{%- endif %}
{%- if "dispatch_distribution_data" in data %}

```{list-table} Dispatch distribution comparison
---
header-rows: 1
align: center
---
* - Model
{%- for h in data["dispatch_distribution_data"]["header"] %}
  - {{ h }}
{%- endfor %}
{%- for i, name in enumerate(data["model_names"]) %}
* - {{ name }}
{%- for h in data["dispatch_distribution_data"]["header"] %}
  - {{ data["dispatch_distribution_data"][h][i] }}
{%- endfor %}
{%- endfor %}
```

{%- endif %}
{%- if "affinities" in data %}

````{admonition} Comparison of statistics of static and dynamic dispatches
---
collapsible: True
---

{%- for dev in data["devices"] %}
```{list-table} Details of static dispatches for {{ data["device_names"].get(dev, dev.capitalize()) }}
---
header-rows: 1
align: center
---
* - Model name
  - Number of tensor elements
  - Estimated work elements
  - Logical data size [B]
  - Estimated work size [B]
{%- for name, aff in zip(data["model_names"], data["affinities"]) %}
* - {{ name }}
  - {{ aff[dev]["static-elements-count"] }}
  - {{ aff[dev]["static-work-elements"] }}
  - {{ aff[dev]["static-data-size-bytes"] }}
  - {{ aff[dev]["static-work-bytes"] }}
{%- endfor %}
```
{%- endfor %}

{%- for dev in data["devices"] %}
```{list-table} Details of dynamic dispatches for {{ data["device_names"].get(dev, dev.capitalize()) }}
---
header-rows: 1
align: center
---
* - Model name
  - Has dynamic elements
  - Has dynamic size
{%- for name, aff in zip(data["model_names"], data["affinities"]) %}
* - {{ name }}
  - {{ aff[dev]["has-dynamic-elements"] }}
  - {{ aff[dev]["has-dynamic-data-size"] }}
{%- endfor %}
```
{%- endfor %}

{%- for dev in data["devices"] %}
```{list-table} Comparison of dispatches types for {{ data["device_names"].get(dev, dev.capitalize()) }}
---
header-rows: 1
align: center
---
* - Model name
  - Fill dispatches
  - Copy dispatches
{%- for name, aff in zip(data["model_names"], data["affinities"]) %}
* - {{ name }}
  - {{ aff[dev]["fill-count"] }}
  - {{ aff[dev]["copy-count"] }}
{%- endfor %}
```
{%- endfor %}

````

{%- endif %}


{%- if "register_allocations_path" in data or "register_allocations_data" in data %}
### Register allocations

{%- for regalloc_type in data["regalloc_types"] %}

#### {{ regalloc_type }}

{%- if "register_allocations_path" in data and regalloc_type in data["register_allocations_path"] %}

```{figure} {{ data["register_allocations_path"][regalloc_type] }}
---
name: {{basename}}_{{regalloc_type}}_register_allocation_path
alt: Comparison of register allocations
align: center
---

Register allocations for {{ regalloc_type }}
```

{%- endif %}
{%- if "register_allocations_data" in data and regalloc_type in data["register_allocations_data"] %}
{%- set regs = data["register_allocations_data"][regalloc_type] %}


```{list-table} Allocations for dispatches
---
header-rows: 1
align: center
---
* - Dispatch
  - Model
  - Number of spills
  - Has scalar spills
  - Number of reloads
  - Number of used vector registers
  - Vector registers
{%- set first_model = regs.items().__iter__().__next__()[0][1] %}
{%- for (dispatch_name, model), dispatch in regs.items() %}
* - {{ ('`' + dispatch_name + '`') if model == first_model else '' }}
  - {{ model }}
  - {{ dispatch.get("vec_spills", "") }}
  - {{ dispatch.get("has_scalar_spills", "") }}
  - {{ dispatch.get("vec_reloads", "") }}
  - {{ dispatch.get("global_vector_registers_count", "") }}
  - {{ ("`" + "`, `".join(dispatch["global_vector_registers"]) + "`") if "global_vector_registers" in dispatch else "" }}
{%- endfor %}
{%- if "register_allocation_summary" in data %}
{%- for idx, (model, d) in enumerate(data["register_allocation_summary"][regalloc_type].items()) %}
* - {{ "Summary" if idx == 0 else "" }}
  - {{ model }}
  - {{ d["total_spills"] }}
  - {{ d["has_scalar_spills"] }}
  - {{ d["total_reloads"] }}
  - {{ d["used_vector_registers_count"] }}
  - `{{ "`, `".join(d["used_vector_registers"]) }}`
{%- endfor %}
{%- endif %}
```

{%- endif %}
{%- endfor %}
{%- endif %}

