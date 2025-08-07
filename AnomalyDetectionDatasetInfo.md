# Module description
Dataset wrapper for anomaly detection in time series.

# Class AnomalyDetectionDataset

Generic dataset for anomaly detection in time series problem.

It reads data from provided CSV file and prepares sequences of data.

CSV file has to follow the schema:

| Timestamp column | Param 1 name     | Param 2 name     | ... | Param N name     | Label          |
|------------------|------------------|------------------|-----|------------------|----------------|
| timestamps       | Numerical values | Numerical values | ... | Numerical values | Integer values |

Kenning automatically discards the timestamp column, as well as the
header row.

The numerical values of parameters are used as signals or data from
sensors, whereas the labels specify anomaly occurrence (values greater
than 0).

Each label describes whether an anomaly has been observed within
<span class="title-ref">window_size</span> previous samples.

This results with final version of dataset where one entry looks like:

| X                                                                     |     |                                                                       | Y                             |
|-----------------------------------------------------------------------|-----|-----------------------------------------------------------------------|-------------------------------|
| Param 1 value from <span class="title-ref">t - window_size + 1</span> | ... | Param N value from <span class="title-ref">t - window_size + 1</span> |                               |
| ...                                                                   | ... | ...                                                                   |                               |
| Param 1 value from <span class="title-ref">t - 1</span>               | ... | Param N value from <span class="title-ref">t - 1</span>               |                               |
| Param 1 value from <span class="title-ref">t</span>                   | ... | Param N value from <span class="title-ref">t</span>                   | 0 (no anomaly) or 1 (anomaly) |


