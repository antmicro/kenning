## Text summarization comparison

{% set basename = data["report_name_simple"] if "model_name" not in data else data["report_name_simple"] + data["model_name"] %}

### Rouge metrics
```{figure} {{data["barplot_rouge_path_comparison"]}}

---
name: {{basename}}_rouge
alt: Rouge scores
align: center
---
Rouge metrics
```
