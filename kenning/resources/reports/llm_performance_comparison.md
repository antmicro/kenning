## Tokens per second comparison

{% set basename = data["report_name_simple"] if "model_name" not in data else data["report_name_simple"] + data["model_name"] %}

### Tokens per second
```{figure} {{data["barplot_tokens_per_second_comparison"]}}

---
name: {{basename}}_tokens_per_second
alt: Tokens per second
align: center
---
Tokens per second
```
