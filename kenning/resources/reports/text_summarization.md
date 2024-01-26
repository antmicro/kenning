## Text summarization metrics{% if data["model_name"] %} for {{data["model_name"]}}{% endif %}

{% set basename = data["report_name_simple"] if "model_name" not in data else data["report_name_simple"] + data["model_name"] %}

### Rouge metrics

```{figure} {{data["barplot_rouge_path"]}}
---
name: {{basename}}_rouge
alt: Rouge scores
align: center
---
Rouge metrics
```

{%- if 'example_predictions' in data %}
### Example summarizations
{%- for prediction in data['example_predictions'][:-1] %}

**target** - {{ prediction['target'] }}

**prediction** - {{ prediction['prediction'] }}

---
{%- endfor %}

**target** - {{ data['example_predictions'][-1]['target'] }}

**prediction** - {{ data['example_predictions'][-1]['prediction'] }}

{%- endif %}

