# Kenning measurements

Kenning measurements are a set of information describing the compilation and evaluation process happending in Kenning.

It contains such information as:

* Classes used to construct the optimization/runtime pipeline, along with their parameters,
* JSON scenario used in the run,
* Command used to run the scenario,
* Versions of used Python modules,
* Performance measurements, such as CPU usage, GPU usage,
* Quality measurements, such as predictions, ground truth, confusion matrix

All information is stored in JSON format.

## Performance metrics

While quality measurements are problem-specific (collected in `evaluate` method of the [](dataset-api) class), the performance metrics are common across devices and applications.

Metrics are collected with certain prefix `<prefix>`, indicating the scope of computations.
There are:

* `<prefix>_timestamp` - gives the timestamp of collecting the measurements in `ns`.
* `<prefix>_cpus_percent` - gives per-core CPU utilization in % in a form of list of lists.
  They are % of per-CPU usages for every timestamp.
* `<prefix>_mem_percent` - gives overall memory usage in %.
* `<prefix>_gpu_utilization` - gives overall GPU utilization in % (only works on platforms with NVIDIA GPUs and NVIDIA Jetson embedded devices).
* `<prefix>_gpu_mem_utilization` - gives GPU memory utilization in % (only works on platforms with NVIDIA GPUs and NVIDIA Jetson embedded devices).
