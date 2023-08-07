# Displaying information about available classes

Kenning provides several scripts for providing information about classes 
(such as [](dataset-api), [](modelwrapper-api), [](optimizer-api)) that are available in the project.


## Kenning list

`kenning list` lists all available classes, grouping them by the base class (group of modules).

The script can be executed as follows:

```bash
kenning list
```

```
Optimizers (in kenning.compilers):

    kenning.compilers.nni_pruning.NNIPruningOptimizer
    kenning.compilers.onnx.ONNXCompiler
    kenning.compilers.tensorflow_pruning.TensorFlowPruningOptimizer
    kenning.compilers.model_inserter.ModelInserter
    kenning.compilers.tvm.TVMCompiler
    kenning.compilers.iree.IREECompiler
    kenning.compilers.tensorflow_clustering.TensorFlowClusteringOptimizer
    kenning.compilers.tflite.TFLiteCompiler

Datasets (in kenning.datasets):

    kenning.datasets.pet_dataset.PetDataset
    kenning.datasets.visual_wake_words_dataset.VisualWakeWordsDataset
    kenning.datasets.random_dataset.RandomizedDetectionSegmentationDataset
    kenning.datasets.open_images_dataset.OpenImagesDatasetV6
    kenning.datasets.random_dataset.RandomizedClassificationDataset
    kenning.datasets.common_voice_dataset.CommonVoiceDataset
    kenning.datasets.magic_wand_dataset.MagicWandDataset
    kenning.datasets.coco_dataset.COCODataset2017
    kenning.datasets.imagenet_dataset.ImageNetDataset

Modelwrappers (in kenning.modelwrappers):

    kenning.modelwrappers.instance_segmentation.yolact.YOLACT
    kenning.modelwrappers.classification.tflite_magic_wand.MagicWandModelWrapper
    kenning.modelwrappers.classification.pytorch_pet_dataset.PyTorchPetDatasetMobileNetV2
    kenning.modelwrappers.detectors.darknet_coco.TVMDarknetCOCOYOLOV3
    kenning.modelwrappers.instance_segmentation.yolact.YOLACTWithPostprocessing
    kenning.modelwrappers.classification.tensorflow_imagenet.TensorFlowImageNet
    kenning.modelwrappers.classification.tflite_person_detection.PersonDetectionModelWrapper
    kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2
    kenning.modelwrappers.instance_segmentation.pytorch_coco.PyTorchCOCOMaskRCNN
    kenning.modelwrappers.detectors.yolov4.ONNXYOLOV4

...

```

The output of the command can be limited by proving one or more positional arguments representing the 
groups of modules: `optimizers`, `runners`, `dataproviders`, `datasets`,
`modelwrappers`, `onnxconversions`, `outputcollectors`, `runtimes`.

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

More verbose information is available by using `-v`, `-vv` flags - dependencies,
descriptions and other information will be displayed for each class.

## Kenning info

`kenning info` can be used to display more detailed information about a 
particular class. This information is especially useful when creating JSON scenario 
configurations. The command displays the following:

* docstrings
* dependencies, along with the information whether they are available in the 
  current Python environment
* supported input and output formats
* arguments structure used in JSON

Let's consider a scenario where we want to compose a Kenning flow utilizing a 
YOLOv4 ModelWrapper. 
This will display all the necessary information about the class:

```bash
kenning info kenning.modelwrappers.detectors.yolov4.ONNXYOLOV4
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
For example, some values in the input/output specification are expressions, because 
the command did not import or evaluate any values. 
This is done to allow for missing dependencies.
```

### Loading a class with arguments

To gain access to more detailed information, the `--load-class-with-args` 
argument can be used. Provided that all dependencies are satisfied and the required.

From the example above, the ONNXYOLOV4 configuration specifies that the `model_path` argument is required. 
All dependencies are available as there is no warning message.

To load a class with arguments, run this command:

```bash
kenning info kenning.modelwrappers.detectors.yolov4.ONNXYOLOV4 \
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

