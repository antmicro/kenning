## Inference performance metrics{% if data["model_name"] %} for {{data["model_name"]}}{% endif %}

### Tokens per second

```{list-table} Tokens per second
---
header-rows: 1
align: center
---
* - Statistic
  - Value
* - Mean
  - **{{'%.3f' %  data['tokens_per_second_mean'] }}**
* - Standard deviation
  - **{{'%.3f' %  data['tokens_per_second_std'] }}**
* - Median
  - **{{'%.3f' %  data['tokens_per_second_median'] }}**
* - Minimum
  - **{{'%.3f' %  data['tokens_per_second_min'] }}**
* - Maximum
  - **{{'%.3f' %  data['tokens_per_second_max'] }}**

```


