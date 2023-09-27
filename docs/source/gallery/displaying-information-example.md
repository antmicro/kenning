# Displaying information about available classes

Kenning provides several scripts for assessing information about classes (such as [](dataset-api), [](modelwrapper-api), [](optimizer-api)) that are available in the project.

Firstly, make sure that Kenning is installed:
```bash
pip install "kenning @ git+https://github.com/antmicro/kenning.git"
```

## Kenning list

`kenning list` lists all available classes, grouping them by the base class (group of modules).

The script can be executed as follows:

```bash
kenning list
```

```
Optimizers (in kenning.optimizers):

    kenning.optimizers.onnx.ONNXCompiler
    kenning.optimizers.tensorflow_optimizers.TensorFlowOptimizer
    kenning.optimizers.tvm.TVMCompiler
    kenning.optimizers.iree.IREECompiler
    kenning.optimizers.tensorflow_pruning.TensorFlowPruningOptimizer
    kenning.optimizers.tensorflow_clustering.TensorFlowClusteringOptimizer
    kenning.optimizers.nni_pruning.NNIPruningOptimizer
    kenning.optimizers.tflite.TFLiteCompiler
    kenning.optimizers.model_inserter.ModelInserter

Datasets (in kenning.datasets):

    kenning.datasets.random_dataset.RandomizedClassificationDataset
    kenning.datasets.coco_dataset.COCODataset2017
    kenning.datasets.open_images_dataset.OpenImagesDatasetV6
    kenning.datasets.helpers.detection_and_segmentation.ObjectDetectionSegmentationDataset
    kenning.datasets.magic_wand_dataset.MagicWandDataset
    kenning.datasets.common_voice_dataset.CommonVoiceDataset
    kenning.datasets.pet_dataset.PetDataset
    kenning.datasets.random_dataset.RandomizedDetectionSegmentationDataset
    kenning.datasets.imagenet_dataset.ImageNetDataset
    kenning.datasets.visual_wake_words_dataset.VisualWakeWordsDataset

Modelwrappers (in kenning.modelwrappers):

    kenning.modelwrappers.instance_segmentation.pytorch_coco.PyTorchCOCOMaskRCNN
    kenning.modelwrappers.object_detection.darknet_coco.TVMDarknetCOCOYOLOV3
    kenning.modelwrappers.instance_segmentation.yolact.YOLACTWithPostprocessing
    kenning.modelwrappers.classification.tensorflow_imagenet.TensorFlowImageNet
    kenning.modelwrappers.instance_segmentation.yolact.YOLACTWrapper
    kenning.modelwrappers.object_detection.yolo_wrapper.YOLOWrapper
    kenning.modelwrappers.frameworks.tensorflow.TensorFlowWrapper
    kenning.modelwrappers.classification.tflite_magic_wand.MagicWandModelWrapper
    kenning.modelwrappers.classification.tflite_person_detection.PersonDetectionModelWrapper
    kenning.modelwrappers.instance_segmentation.yolact.YOLACT
    kenning.modelwrappers.frameworks.pytorch.PyTorchWrapper
    kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2
    kenning.modelwrappers.object_detection.yolov4.ONNXYOLOV4
    kenning.modelwrappers.classification.pytorch_pet_dataset.PyTorchPetDatasetMobileNetV2

...

```

The output of the command can be limited by providing one or more positional arguments representing the groups of modules: `optimizers`, `runners`, `dataproviders`, `datasets`, `modelwrappers`, `onnxconversions`, `outputcollectors`, `runtimes`.

The command can be used to only list the available runtimes:

```bash
kenning list runtimes
```

```
Runtimes (in kenning.runtimes):

    kenning.runtimes.iree.IREERuntime
    kenning.runtimes.tflite.TFLiteRuntime
    kenning.runtimes.pytorch.PyTorchRuntime
    kenning.runtimes.tvm.TVMRuntime
    kenning.runtimes.onnx.ONNXRuntime
    kenning.runtimes.renode.RenodeRuntime
```

More verbose information is available by using `-v`, `-vv` flags - dependencies, descriptions and other information will be displayed for each class.

## Kenning info

`kenning info` can be used to display more detailed information about a particular class.
This information is especially useful when creating JSON scenario configurations.
The command displays the following:

* docstrings
* dependencies, along with the information whether they are available in the current Python environment
* supported input and output formats
* arguments structure used in JSON

Let's consider a scenario where we want to compose a Kenning flow utilizing a YOLOv4 ModelWrapper.
This will display all the necessary information about the class:

```bash
kenning info kenning.modelwrappers.object_detection.yolov4.ONNXYOLOV4
```

```
Class: ONNXYOLOV4

Input/output specification:
* input
  * shape: (1, 3, keyparams['width'], keyparams['height'])
  * dtype: float32
* output
  * shape: (1, 255, (keyparams['width'] // (8 * (2 ** 0))), (keyparams['height'] // (8 * (2 ** 0))))
  * dtype: float32
* output.3
  * shape: (1, 255, (keyparams['width'] // (8 * (2 ** 1))), (keyparams['height'] // (8 * (2 ** 1))))
  * dtype: float32
* output.7
  * shape: (1, 255, (keyparams['width'] // (8 * (2 ** 2))), (keyparams['height'] // (8 * (2 ** 2))))
  * dtype: float32
* detection_output
  * type: List[DetectObject]

Dependencies:
* torch
* numpy
* onnx
* torch.nn.functional

Arguments specification:
* classes
  * argparse_name: --classes
  * convert-type: builtins.str
  * type
    * string
  * description: File containing Open Images class IDs and class names in CSV format to use (can be generated using kenning.scenarios.open_images_classes_extractor) or class type
  * default: coco
* model_path
  * argparse_name: --model-path
  * convert-type: kenning.utils.resource_manager.ResourceURI
  * type
    * string
  * description: Path to the model
  * required: True
```

```{note}
By default, the command only performs static code analysis.
For example, some values in the input/output specification are expressions, because the command did not import or evaluate any values.
This is done to allow for missing dependencies.
```

### Loading a class with arguments

To gain access to more detailed information, the `--load-class-with-args` argument can be used.
Provided that all dependencies are satisfied and the required.

From the example above, the ONNXYOLOV4 configuration specifies that the `model_path` argument is required.
All dependencies are available as there is no warning message.

To load a class with arguments, run this command:

```bash
kenning info kenning.modelwrappers.object_detection.yolov4.ONNXYOLOV4 \
  --load-class-with-args \
  --model-path kenning:///models/detection/yolov4.onnx
```

```
Class: ONNXYOLOV4

Input/output specification:
* input
  * shape: (1, 3, 608, 608)
  * dtype: float32
* output
  * shape: (1, 255, 76, 76)
  * dtype: float32
* output.3
  * shape: (1, 255, 38, 38)
  * dtype: float32
* output.7
  * shape: (1, 255, 19, 19)
  * dtype: float32
* detection_output
  * type: List[DetectObject]

Dependencies:
* onnx
* numpy
* torch.nn.functional
* torch

Arguments specification:
* classes
  * argparse_name: --classes
  * convert-type: builtins.str
  * type
    * string
  * description: File containing Open Images class IDs and class names in CSV format to use (can be generated using kenning.scenarios.open_images_classes_extractor) or class type
  * default: coco
* model_path
  * argparse_name: --model-path
  * convert-type: kenning.utils.resource_manager.ResourceURI
  * type
    * string
  * description: Path to the model
  * required: True
```

