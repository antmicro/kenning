Object detection metrics{% if data["modelname"] %} for {{data["modelname"]}}{% endif %}
------------------------{% if data["modelname"] %}{{'-' * (' for ' + data["modelname"])|length}}{% endif %}

{% set basename = data["reportname"] if "modelname" not in data else data["reportname"] + data["modelname"] %}
.. figure:: {{data["curvepath"]}}
    :name: {{basename}}_recall_precision_curves
    :alt: Recall-Precision curves
    :align: center

    Per-Class Recall-Precision curves

.. figure:: {{data["gradientpath"]}}
    :name: {{basename}}_recall_precision_gradients
    :alt: Per-Class precision gradients
    :align: center

    Per-Class precision gradients

.. figure:: {{data["mappath"]}}
    :name: {{basename}}_map
    :alt: mAP values depending on threshold
    :align: center

    mAP values depending on threshold

* *Mean Average Precision* for threshold 0.5: {{data['mAP']}}
* Best *Mean Average Precision* occurs at threshold {{data['max_mAP_index']}}  and it is: {{data['max_mAP']}}

.. figure:: {{data["tpioupath"]}}
    :name: {{basename}}_tpiou
    :alt: Per-Class mean IoU values for correctly labeled objects
    :align: center

    Per-Class mean IoU values for correctly labeled objects

.. figure:: {{data["iouhistpath"]}}
    :name: {{basename}}_iouhist
    :alt: Histogram of IoU values for correctly labeled objects
    :align: center

    Histogram of IoU values for correctly labeled objects


