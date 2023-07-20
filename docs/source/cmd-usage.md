# Using Kenning via command line arguments

{{projecturl}} provides several scripts for training, compilation and benchmarking of deep learning models on various target hardware.
The executable scripts are present in the [kenning.scenarios module](https://github.com/antmicro/kenning/tree/main/kenning/scenarios).
Sample bash scripts using the scenarios are located in the [scripts directory in the repository](https://github.com/antmicro/kenning/tree/main/scripts).

Runnable scripts in scenarios require implemented classes to be provided from the `kenning.core` module to perform such actions as in-framework inference, model training, model compilation and model benchmarking on target.

## Command-line arguments for classes

Each class ([](dataset-api), [](modelwrapper-api), [](optimizer-api) and other) provided to the runnable scripts in scenarios can provide command-line arguments that configure the work of an object of the given class.

Each class in `kenning.core` implements `form_argparse` and `from_argparse` methods.
The former creates an `argparse` group for a given class with its parameters.
The latter takes the arguments parsed by `argparse` and returns the object of a class.

## Model training

`kenning.scenarios.model_training` performs model training using Kenning's [](modelwrapper-api) and [](dataset-api) objects.
To get the list of training parameters, select the model and training dataset to use (i.e. `TensorFlowPetDatasetMobileNetV2` model and `PetDataset` dataset) and run:

```bash
python -m kenning.scenarios.model_training \
    kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    kenning.datasets.pet_dataset.PetDataset \
    -h
```

This will list the possible parameters that can be used to configure the dataset, the model, and the training parameters.
For the above call, the output is as follows:

```bash
positional arguments:
  modelwrappercls       ModelWrapper-based class with inference implementation to import
  datasetcls            Dataset-based class with dataset to import

optional arguments:
  -h, --help            show this help message and exit
  --batch-size BATCH_SIZE
                        The batch size for training
  --learning-rate LEARNING_RATE
                        The learning rate for training
  --num-epochs NUM_EPOCHS
                        Number of epochs to train for
  --logdir LOGDIR       Path to the training logs directory
  --verbosity {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Verbosity level

Inference model arguments:
  --model-path MODEL_PATH
                        Path to the model

Dataset arguments:
  --dataset-root DATASET_ROOT
                        Path to the dataset directory
  --download-dataset    Downloads the dataset before taking any action
  --inference-batch-size INFERENCE_BATCH_SIZE
                        The batch size for providing the input data
  --classify-by {species,breeds}
                        Determines if classification should be performed by species or by breeds
  --image-memory-layout {NHWC,NCHW}
                        Determines if images should be delivered in NHWC or NCHW format
```

```{note}
The list of options depends on [](modelwrapper-api) and [](dataset-api).
```

At the end, the training can be configured as follows:

```bash
python -m kenning.scenarios.model_training \
    kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    kenning.datasets.pet_dataset.PetDataset \
    --logdir build/logs \
    --dataset-root build/pet-dataset \
    --model-path build/trained-model.h5 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --num-epochs 50
```

This will train the model with a 0.0001 learning rate and batch size 32 for 50 epochs.
The trained model will be saved as `build/trained-model.h5`.

## In-framework inference performance measurements

The `kenning.scenarios.inference_performance` script runs inference on a given model in a framework it was trained on.
It requires you to provide:

* a [](modelwrapper-api)-based object wrapping the model to be tested,
* a [](dataset-api)-based object wrapping the dataset applicable to the model,
* a path to the output JSON file with performance and quality metrics gathered during inference by the [](measurements-api) object.

The example call for the method is as follows:

```bash
python -m kenning.scenarios.inference_tester \
    --modelwrapper-cls kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    --dataset-cls kenning.datasets.pet_dataset.PetDataset \
    --measurements build/tensorflow_pet_dataset_mobilenetv2.json \
    --model-path kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --dataset-root build/pet-dataset/ \
    --download-dataset \
    --run-benchmarks-only
```

The script downloads the dataset to the `build/pet-dataset` directory, loads the `tensorflow_pet_dataset_mobilenetv2.h5` model, runs inference on all images from the dataset and collects performance and quality metrics throughout the run.
The performance data stored in the JSON file can be later rendered using [](report-generation).

## ONNX conversion

`kenning.scenarios.onnx_conversion` empirically tests the ONNX conversion for various frameworks and generates a report containing a support matrix.
The matrix tells us if model export to ONNX and model import from ONNX for a given framework and model are supported or not.
The example report with the command call is available in [](./onnx-conversion-support).

`kenning.scenarios.onnx_conversion` requires a list of [](onnxconversion-api) classes that implement model providers and a conversion method.
For the below, call:

```bash
python -m kenning.scenarios.onnx_conversion \
    build/models-directory \
    build/onnx-support.rst \
    --converters-list \
        kenning.onnxconverters.pytorch.PyTorchONNXConversion \
        kenning.onnxconverters.tensorflow.TensorFlowONNXConversion \
        kenning.onnxconverters.mxnet.MXNetONNXConversion
```

The conversion is tested for three frameworks - PyTorch, TensorFlow and MXNet.
The successfully converted ONNX models are stored in the `build/models-directory`.
The final RST file with the report is stored in the `build/onnx-support.rst` directory.

## Testing inference on target hardware

The `kenning.scenarios.inference_tester` and `kenning.scenarios.inference_server` are used for inference testing on target hardware.
The `inference_tester` loads the dataset and the model, compiles the model and runs inference either locally or remotely using `inference_server`.

The `inference_server` receives the model and input data, and sends output data and statistics.

Both `inference_tester` and `inference_server` require [](runtime-api) to determine the model execution flow.
Both scripts communicate using the communication protocol implemented in the [](runtimeprotocol-api).

At the end, the `inference_tester` returns the benchmark data in the form of a JSON file extracted from the [](measurements-api) object.

The `kenning.scenarios.inference_tester` requires:

* a [](modelwrapper-api)-based class that implements model loading, I/O processing and optionally model conversion to ONNX format,
* a [](runtime-api)-based class that implements data processing and the inference method for the compiled model on the target hardware,
* a [](dataset-api)-based class that implements data sample fetching and model evaluation,
* a path to the output JSON file with performance and quality metrics.

An [](optimizer-api)-based class can be provided to compile the model for a given target if needed.

Optionally, it requires a [](runtimeprotocol-api)-based class when running remotely to communicate with the `kenning.scenarios.inference_server`.

To print the list of required arguments, run:

```bash
python3 -m kenning.scenarios.inference_tester \
    --modelwrapper-cls kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    --runtime-cls kenning.runtimes.tvm.TVMRuntime \
    --dataset-cls kenning.datasets.pet_dataset.PetDataset \
    --modelcompiler-cls kenning.compilers.tvm.TVMCompiler \
    --protocol-cls kenning.runtimeprotocols.network.NetworkProtocol \
    --measurements ''
    -h
```

With the above classes, the help can look as follows:

```bash
optional arguments:
  -h, --help            show this help message and exit

ModelWrapper arguments:
  --model-path MODEL_PATH
                        Path to the model

PetDataset arguments:
  --classify-by {species,breeds}
                        Determines if classification should be performed by species or by breeds
  --image-memory-layout {NHWC,NCHW}
                        Determines if images should be delivered in NHWC or NCHW format

Dataset arguments:
  --dataset-root DATASET_ROOT
                        Path to the dataset directory
  --inference-batch-size INFERENCE_BATCH_SIZE
                        The batch size for providing the input data
  --download-dataset    Downloads the dataset before taking any action. If the dataset files are already downloaded
                        then they are not downloaded again
  --force-download-dataset
                        Forces dataset download
  --external-calibration-dataset EXTERNAL_CALIBRATION_DATASET
                        Path to the directory with the external calibration dataset
  --split-fraction-test SPLIT_FRACTION_TEST
                        Default fraction of data to leave for model testing
  --split-fraction-val SPLIT_FRACTION_VAL
                        Default fraction of data to leave for model valdiation
  --split-seed SPLIT_SEED
                        Default seed used for dataset split

TVMRuntime arguments:
  --save-model-path SAVE_MODEL_PATH
                        Path where the model will be uploaded
  --target-device-context {llvm,stackvm,cpu,c,test,hybrid,composite,cuda,nvptx,cl,opencl,sdaccel,aocl,aocl_sw_emu,vulkan,metal,vpi,rocm,ext_dev,hexagon,webgpu}
                        What accelerator should be used on target device
  --target-device-context-id TARGET_DEVICE_CONTEXT_ID
                        ID of the device to run the inference on
  --runtime-use-vm      At runtime use the TVM Relay VirtualMachine

Runtime arguments:
  --disable-performance-measurements
                        Disable collection and processing of performance metrics

NetworkProtocol arguments:
  --host HOST           The address to the target device
  --port PORT           The port for the target device

BytesBasedProtocol arguments:
  --packet-size PACKET_SIZE
                        The maximum size of the received packets, in bytes.
  --endianness {big,little}
                        The endianness of data to transfer
```

The `kenning.scenarios.inference_server` requires only:

* a [](runtimeprotocol-api)-based class for the implementation of the communication,
* a [](runtime-api)-based class for the implementation of runtime routines on device.

Both classes may require some additional arguments that can be listed with the `-h` flag.

An example script for the `inference_tester` is:

```bash
python -m kenning.scenarios.inference_tester \
    --modelwrapper-cls kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    --runtime-cls kenning.runtimes.tflite.TFLiteRuntime \
    --dataset-cls kenning.datasets.pet_dataset.PetDataset \
    --measurements ./build/google-coral-devboard-tflite-tensorflow.json \
    --compiler-cls kenning.compilers.tflite.TFLiteCompiler \
    --protocol-cls kenning.runtimeprotocols.network.NetworkProtocol \
    --model-path ./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework keras \
    --target "edgetpu" \
    --compiled-model-path build/compiled-model.tflite \
    --inference-input-type int8 \
    --inference-output-type int8 \
    --host 192.168.188.35 \
    --port 12345 \
    --packet-size 32768 \
    --save-model-path /home/mendel/compiled-model.tflite \
    --dataset-root build/pet-dataset \
    --inference-batch-size 1 \
    --verbosity INFO
```

The above runs with the following `inference_server` setup:

```bash
python -m kenning.scenarios.inference_server \
    --protocol-cls kenning.runtimeprotocols.network.NetworkProtocol \
    --runtime-cls kenning.runtimes.tflite.TFLiteRuntime \
    --host 0.0.0.0 \
    --port 12345 \
    --packet-size 32768 \
    --save-model-path /home/mendel/compiled-model.tflite \
    --delegates-list libedgetpu.so.1 \
    --verbosity INFO
```

```{note}
This run was tested on a Google Coral Devboard device.
```

`kenning.scenarios.inference_tester` can be also executed locally - in this case, the `--protocol-cls` argument can be skipped.
The example call is as follows:

```bash
python3 -m kenning.scenarios.inference_tester \
    kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
    kenning.runtimes.tvm.TVMRuntime \
    kenning.datasets.pet_dataset.PetDataset \
    ./build/local-cpu-tvm-tensorflow-classification.json \
    --modelcompiler-cls kenning.compilers.tvm.TVMCompiler \
    --model-path ./kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
    --model-framework keras \
    --target "llvm" \
    --compiled-model-path ./build/compiled-model.tar \
    --opt-level 3 \
    --save-model-path ./build/compiled-model.tar \
    --target-device-context cpu \
    --dataset-root ./build/pet-dataset/ \
    --inference-batch-size 1 \
    --download-dataset \
    --verbosity INFO
```

```{note}
For more examples of running `inference_tester` and `inference_server`, check the [kenning/scripts](https://github.com/antmicro/kenning/tree/main/scripts) directory.
Directories with scripts for client and server calls for various target devices, deep learning frameworks and compilation frameworks can be found in the [kenning/scripts/edge-runtimes](https://github.com/antmicro/kenning/tree/main/scripts/edge-runtimes) directory.
```

## Running inference

`kenning.scenarios.inference_runner` is used to run inference locally on a pre-compiled model.

`kenning.scenarios.inference_runner` requires:

* a [](modelwrapper-api)-based class that performs I/O processing specific to the model,
* a [](runtime-api)-based class that runs inference on target using the compiled model,
* a [](dataprovider-api)-based class that implements fetching of data samples from various sources,
* a list of [](outputcollector-api)-based classes that implement output processing for the specific use case.

To print the list of required arguments, run:

```bash
python3 -m kenning.scenarios.inference_runner \
    kenning.modelwrappers.detectors.darknet_coco.TVMDarknetCOCOYOLOV3 \
    kenning.runtimes.tvm.TVMRuntime \
    kenning.dataproviders.camera_dataprovider.CameraDataProvider \
     --output-collectors kenning.outputcollectors.name_printer.NamePrinter \
    -h
```

With the above classes, the help can look as follows:

```bash
positional arguments:
  modelwrappercls       ModelWrapper-based class with inference implementation to import
  runtimecls            Runtime-based class with the implementation of model runtime
  dataprovidercls       DataProvider-based class used for providing data
optional arguments:
  -h, --help            show this help message and exit
  --output-collectors OUTPUT_COLLECTORS [OUTPUT_COLLECTORS ...]
                        List to the OutputCollector-based classes where the results will be passed
  --verbosity {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Verbosity level
Inference model arguments:
  --model-path MODEL_PATH
                        Path to the model
  --classes CLASSES     File containing Open Images class IDs and class names in CSV format to use (can be generated using
                        kenning.scenarios.open_images_classes_extractor) or class type
Runtime arguments:
  --disable-performance-measurements
                        Disable collection and processing of performance metrics
  --save-model-path SAVE_MODEL_PATH
                        Path where the model will be uploaded
  --target-device-context {llvm,stackvm,cpu,c,cuda,nvptx,cl,opencl,aocl,aocl_sw_emu,sdaccel,vulkan,metal,vpi,rocm,ext_dev,hexagon,webgpu}
                        What accelerator should be used on target device
  --target-device-context-id TARGET_DEVICE_CONTEXT_ID
                        ID of the device to run the inference on
  --input-dtype INPUT_DTYPE
                        Type of input tensor elements
  --runtime-use-vm      At runtime use the TVM Relay VirtualMachine
  --use-json-at-output  Encode outputs of models into a JSON file with base64-encoded arrays
DataProvider arguments:
  --video-file-path VIDEO_FILE_PATH
                        Video file path (for cameras, use /dev/videoX where X is the device ID eg. /dev/video0)
  --image-memory-layout {NHWC,NCHW}
                        Determines if images should be delivered in NHWC or NCHW format
  --image-width IMAGE_WIDTH
                        Determines the width of the image for the model
  --image-height IMAGE_HEIGHT
                        Determines the height of the image for the model
OutputCollector arguments:
  --print-type {detector,classificator}
                        What is the type of model that will input data to the NamePrinter
```

An example script for `inference_runner`:

```bash
python3 -m kenning.scenarios.inference_runner \
    kenning.modelwrappers.detectors.darknet_coco.TVMDarknetCOCOYOLOV3 \
    kenning.runtimes.tvm.TVMRuntime \
    kenning.dataproviders.camera_dataprovider.CameraDataProvider \
    --output-collectors kenning.outputcollectors.detection_visualizer.DetectionVisualizer kenning.outputcollectors.name_printer.NamePrinter \
    --disable-performance-measurements \
    --model-path ./kenning/resources/models/detection/yolov3.weights \
    --save-model-path ../compiled-model.tar \
    --target-device-context "cuda" \
    --verbosity INFO \
    --video-file-path /dev/video0
```

(report-generation)=

## Generating performance reports

`kenning.scenarios.inference_performance` and `kenning.scenarios.inference_tester` return a JSON file as the result of benchmarks.
They contain both performance metrics data, and quality metrics data.

The data from JSON files can be analyzed, processed and visualized by the `kenning.scenarios.render_report` script.
This script parses the information in JSON files and returns an RST file with the report, along with visualizations.

It requires:

* a JSON file with benchmark data,
* a report name for use in the RST file and for creating Sphinx refs to figures,
* an RST output file name,
* `--root-dir` specifying the root directory of the Sphinx documentation where the RST file will be embedded (it is used to compute relative paths),
* `--img-dir` specifying the path where the figures should be saved,
* `--report-types`, which is a list describing the types the report falls into.

An example call and the resulting RST file can be observed in [](sample-report).

As for now, the available report types are:

* `performance` - this is the most common report type that renders information about overall inference performance metrics, such as inference time, CPU usage, RAM usage, or GPU utilization,
* `classification` - this report is specific to the classification task, it renders classification-specific quality figures and metrics, such as confusion matrices, accuracy, precision, G-mean,
* `detection` - this report is specific to the detection task, it renders detection-specific quality figures and metrics, such as recall-precision curves or mean average precision.

## Displaying information about available classes

`kenning.scenarios.list_classes` and `kenning.scenarios.class_info` provide useful information about classes and can help in creating JSON scenarios.

`kenning.scenarios.list_classes` will list all available classes by default, though the output can be limited by providing positional arguments representing groups of modules: `optimizers`, `runners`, `dataproviders`, `datasets`, `modelwrappers`, `onnxconversions`, `outputcollectors`, `runtimes`.
The amount of information displayed can be controlled using `-v` and `-vv` flags.

To print available arguments run `python -m kenning.scenarios.list_classes -h`.

`kenning.scenarios.class_info` provides information about a class given in an argument. More precisely, it will display:

* module and class docstrings,
* dependencies along with the information whether they are available in the current python environment,
* supported input and output formats,
* arguments structure used in JSON configurations.

The script uses a module-like path to the file (e.g. `kenning.runtimes.tflite`), but optionally a class can be specified by adding it to the path like so: `kenning.runtimes.tflite.TFLiteRuntime`. 
To get more detailed information, an optional `--load-class-with-args` argument can be passed. This needs all required class arguments to be provided, and that all dependencies are available.

For more detail, check `python -m kenning.scenarios.class_info -h`.
