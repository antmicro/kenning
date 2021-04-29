Deep Learning deployment stack
==============================

This chapter lists and describes the typical actions performed on deep learning models before deployment on target devices.

From training to deployment
---------------------------

The deep learning application deployed on IoT devices usually goes through the following process:

* the dataset is prepared for deep learning process,
* The evaluation metrics are specified based on a given dataset and outputs,
* the data in the dataset goes through analysis, the data loaders that perform the preprocessing are implemented,
* the deep learning model is either designed from scratch, or the baseline is selected from the wide selection of existing pre-trained models for the given deep learning application (classification, detection, semantic segmentation, instance segmentation, ...) and adjusted to the scenario of the application,
* along with the deep learning model, the loss function and learning algorithm is specified,
* the model is trained, evaluated and improved,
* the model is compiled to a representation applicable for a given target,
* the model is executed on a target device.

Dataset preparation
-------------------

If the model is not available, or is trained for a different use case, the training of the model is required.

Each model requires a dataset - a set of samples of inputs (audio signals, images, video sequences, OCT images, other sensors) and usually also  outputs (association to class or classes, object location, object mask, input description).
The dataset is usually subdivided into:

* training dataset - the largest subset that is used in the model training,
* validation dataset - the relatively small set that is used to verify the model performance after each training epoch (the metrics and loss function values demonstrate if there is any overfitting during the training process),
* test dataset - the subset that acts as the final evaluation of the trained model.

It is required that the test dataset is mutually exclusive from the training dataset so the evaluation results are not biased in any way.

Datasets can be either designed from scratch or found in:

* `Kaggle datasets <https://www.kaggle.com>`_,
* `Google Dataset Search <https://datasetsearch.research.google.com>`_.

Model preparation and training
------------------------------

Currently, the most popular approach is to find existing model that fits a given problem and perform transfer learning to adapt the model to one's needs.
In transfer learning the existing model is slightly modified in its final layers to adapt to a new problem, and the last layers of the model are trained using the training dataset.
In the end, some additional layers are unfreezed and the training is performed on a larger number of parameters with very small learning rate - this process is called fine-tuning.

Transfer learning gives a better starting point for the training process, allows to train a decently performing model with smaller datasets, and reduces the time required to train the model.
The intuition behind this is that there are lots of common features between various objects in the real-life environments, and the features learned on one deep learning scenario can be reused in another scenario.

After selecting the model, the adequate preprocessing of data inputs needs to be provided in order to perform valid training.
The input data should be normalized and resized to fit input tensor requirements.
In case of the training dataset, especially if it is quite small, applying reasonable data augmentations, like random brightness, contrast, cropping, jitters, rotations can significantly improve the training process and prevent the network from overfitting.

In the end, a proper training procedure needs to be specified.
This includes:

* specifying loss function for the model - the loss function should fit the task.
  Along with the loss function, some weights regularizations can be specified to reduce the chance of overfitting.
* specifying optimizer, like Adam, Adagrad.
  This includes setting properly hyperparameters, or adding schedules and automated routines to set those hyperparameters (i.e. scheduling the learning rate value, or using LR-Finder to set the proper learning rate for the scenario).
* specifying or scheduling the number of epochs, i.e. early stopping can be introduced.
* providing some routines for quality metrics measurements.
* providing some routines for saving the intermediate models during training (periodically, or the best model according to some quality measure).

Model optimization
------------------

The successfully trained model may require some optimizations in order to run on a given IoT hardware.
The optimizations may regard the precision of weights, or the computational representation, or the model structure.

Models are usually trained in FP32 precision or mixed precision (FP32 + FP16, depending on the operator).
Some targets, on the other hand, may significantly benefit from changing the FP32 precision to FP16, INT8 or INT4 precision.
The optimizations here are straightforward for the FP16 precision, but the integer-based quantizations require calibration datasets to reduce the precision without siginificant loss of model's quality.

Other optimizations change the computational representation of the model by i.e. layer fusion, specialized operators for convolutions of a particular shape, and other.

In the end, there are algorithmic optimizations that change the whole model structure, like weight's pruning, conditional computation, model distillation (the current model acts as a teacher that is supposed to improve the quality of a much smaller model).

If the model optimizations are applied, the optimized models should be evaluated using the same metrics as the former model in order to find any quality drops.

Model compilation and deployment
--------------------------------

The deep learning compilers can transform the current representation of the model to:

* a source code for a different programming language, e.g. `Halide <https://halide-lang.org>`_, C, C++, Java that can be later used on a given target,
* a binary with acceleration based on i.e. OpenGL, OpenCL, CUDA, TensorRT, ROCm libraries,
* the binary blob for FPGA-based acceleration,
* other targets.

Those compiled models are optimized to perform as efficiently as possible on a given target hardware.

In the end, the models are deployed on a hardware device.
