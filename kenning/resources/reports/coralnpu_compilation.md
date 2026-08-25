## CoralNPU compilation{% if data["model_name"] %} of {{data["model_name"]}}{% endif %}

{%- if "compilation_duration" in data %}
### Duration of compilation

The compilation took **{{ '{:.3f}'.format(data["compilation_duration"]) }}** seconds.
{%- endif %}

{%- if "affinities" in data %}

### Distribution of dispatches

```{list-table} Dispatches distribution
---
header-rows: 1
align: center
---
* - Device
  - Static dispatches
  - Dynamic dispatches
{%- for name, aff in data["affinities"].items() %}
* - {{ data["device_names"].get(name, name.capitalize()) }}
  - {{ aff["static-dispatch-count"] }}
  - {{ aff["dynamic-dispatch-count"] }}
{%- endfor %}
* - Summary
  - {{ data["affinities_summary"]["static-dispatch-count"] }}
  - {{ data["affinities_summary"]["dynamic-dispatch-count"] }}
```
* Static dispatches details
```{list-table} Static dispatches details
---
header-rows: 1
align: center
---
* - Device
  - Number of tensor elements
  - Estimated work elements
  - Logical data size [B]
  - Estimated work size [B]
{%- for name, aff in data["affinities"].items() %}
* - {{ data["device_names"].get(name, name.capitalize()) }}
  - {{ aff["static-elements-count"] }}
  - {{ aff["static-work-elements"] }}
  - {{ aff["static-data-size-bytes"] }}
  - {{ aff["static-work-bytes"] }}
{%- endfor %}
```
* Dynamic dispatches details
```{list-table} Dynamic dispatches details
---
header-rows: 1
align: center
---
* - Device
  - Has dynamic elements
  - Has dynamic size
{%- for name, aff in data["affinities"].items() %}
* - {{ data["device_names"].get(name, name.capitalize()) }}
  - {{ aff["has-dynamic-elements"] }}
  - {{ aff["has-dynamic-data-size"] }}
{%- endfor %}
```

* Dispatches types
```{list-table} Dispatches types
---
header-rows: 1
align: center
---
* - Device
  - Fill dispatches
  - Copy dispatches
{%- for name, aff in data["affinities"].items() %}
* - {{ data["device_names"].get(name, name.capitalize()) }}
  - {{ aff["fill-count"] }}
  - {{ aff["copy-count"] }}
{%- endfor %}
```

````{admonition} Estimated work metric
---
collapsible: True
---
Estimated work metric is calculated as a sum of products across dispatches:
```{math}
\text{work_metric} = \sum_{d} m_do_d
```
where {math}`d` represents dispatches, {math}`m_d` metric for dispatch {math}`d` (either number of elements or size) and {math}`o_d` OP count for dispatch {math}`d`.
````

{%- endif %}


{%- if "register_allocation" in data %}

### Register allocations

{%- for name, regs in data["register_allocation"].items() %}

#### {{ name }}

```{list-table} Allocations for dispatches
---
header-rows: 1
align: center
---
* - Dispatch
  - Number of spills
  - Has scalar spills
  - Number of reloads
  - Number of used vector registers
  - Vector registers
{%- for dispatch in regs["dispatches"] %}
* - `{{ dispatch["name"] }}`
  - {{ dispatch["vec_spills"] }}
  - {{ dispatch["has_scalar_spills"] }}
  - {{ dispatch["vec_reloads"] }}
  - {{ dispatch["global_vector_registers_count"] }}
  - `{{ "`, `".join(dispatch["global_vector_registers"]) }}`
{%- endfor %}
{%- if "register_allocation_summary" in data %}
* - Summary
  - {{ data["register_allocation_summary"][name]["total_spills"] }}
  - {{ data["register_allocation_summary"][name]["has_scalar_spills"] }}
  - {{ data["register_allocation_summary"][name]["total_reloads"] }}
  - {{ data["register_allocation_summary"][name]["used_vector_registers_count"] }}
  - `{{ "`, `".join(data["register_allocation_summary"][name]["used_vector_registers"]) }}`
{%- endif %}
```

```{admonition} Loops' details
---
collapsible: True
---
{%- for dispatch in regs["dispatches"] %}
* `{{ dispatch["name"] }}` dispatch loops:
{%- for loop_ in dispatch["loops"] %}
{{ "  " * loop_["depth"] }}* Header: `{{ loop_["header"] }}`
Location: `{{ loop_["location"] }}`
Depth: {{ loop_["depth"] }}

{{ "  " * (loop_["depth"] + 2) }}Spills: {{ loop_["vec_spills"] }}
{%- if loop_["has_scalar_spills"] %} (has scalar spills) {%- endif %}

    {%- if loop_["has_scalar_spills"] %}
{{ "  " * (loop_["depth"] + 2) }}(has scalar spills)
    {%- endif %}

{{ "  " * (loop_["depth"] + 2) }}Reloads: {{ loop_["vec_reloads"] }}

{{ "  " * (loop_["depth"] + 2) }}Vector registers (used {{ loop_["vector_registers_used_count"] }} time{{ "s" if loop_["vector_registers_used_count"] > 1 else "" }}): `{{ "`, `".join(loop_["vector_registers_used"]) }}`
{%- endfor %}
{%- endfor %}
```

{%- endfor %}
{%- endif %}


