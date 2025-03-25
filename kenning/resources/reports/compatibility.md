## Compatibility tests

{%- for name in data['names'] %}

{# Define roles to add text style #}
```{role} compatible
```
```{role} incompatible
```
```{role} failed
```
```{role} error
```
```{role} fixed
```

### {{name}}

```{figure} {{data['paths'][name]}}
---
name: {{name}}_compatibilitymatrix
alt: Compatibility matrix
align: center
---
```
* *{compatible}`Compatible`*: **{{ data['stats'][name]['Compatible'] }}**
* *{incompatible}`Incompatible`*: **{{ data['stats'][name]['Incompatible'] }}**
* *{failed}`Failed`*: **{{ data['stats'][name]['Failed'] }}**
* *{error}`Error`*: **{{ data['stats'][name]['Error']}}**
* *{fixed}`Fixed`*: **{{ data['stats'][name]['Fixed']}}**

{%- endfor %}
