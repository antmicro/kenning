Object detection metrics
------------------------

.. figure:: {{data["curvepath"]}}
    :name: {{data["reportname"][0]}}_recall_precision_curves
    :alt: Recall-Precision curves
    :align: center

    Per-Class Recall-Precision curves

.. figure:: {{data["gradientpath"]}}
    :name: {{data["reportname"][0]}}_recall_precision_gradients
    :alt: Per-Class precision gradients
    :align: center

    Per-Class precision gradients

.. figure:: {{data["mappath"]}}
    :name: {{data["reportname"][0]}}_map
    :alt: mAP values depending on threshold
    :align: center

    mAP values depending on threshold

* *Mean Average Precision* for threshold 0.5: {{data['mAP']}}
* Best *Mean Average Precision* occurs at threshold {{data['max_mAP_index']}}  and it is: {{data['max_mAP']}}

.. figure:: {{data["tpioupath"]}}
    :name: {{data["reportname"][0]}}_tpiou
    :alt: Per-Class mean IoU values for correctly labeled objects
    :align: center
   
    Per-Class mean IoU values for correctly labeled objects

.. figure:: {{data["iouhistpath"]}}
    :name: {{data["reportname"][0]}}_iouhist
    :alt: Histogram of IoU values for correctly labeled objects
    :align: center
   
    Histogram of IoU values for correctly labeled objects