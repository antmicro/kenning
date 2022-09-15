Inference quality metrics{% if data["modelname"] %} for {{data["modelname"]}}{% endif %}
-------------------------

.. figure:: {{data["confusionpath"]}}
    :name: {{data["reportname"][0]}}_confusionmatrix
    :alt: Confusion matrix
    :align: center

    Confusion matrix

* *Accuracy*: **{{ accuracy(data['eval_confusion_matrix']) }}**
* *Top-5 accuracy*: **{{ data['top_5_count'] / data['total'] }}**
* *Mean precision*: **{{ mean_precision(data['eval_confusion_matrix']) }}**
* *Mean sensitivity*: **{{ mean_sensitivity(data['eval_confusion_matrix']) }}**
* *G-mean*: **{{ g_mean(data['eval_confusion_matrix']) }}**
