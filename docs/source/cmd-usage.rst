Using Kenning from command-line
===============================

|projecturl| provides several scripts for training, compiling and benchmarking deep learning models on various target hardware.
The executable scripts are present in the `kenning.scenarios module <https://github.com/antmicro/kenning/tree/master/kenning/scenarios>`_.
The sample bash scripts using the scenarios are located in the `scripts directory in the repository <https://github.com/antmicro/kenning/tree/master/scripts>`_.

Runnable scripts in scenarios require providing implemented classes from ``kenning.core`` module to perform such actions as in-framework inference, model training, model compilation and model benchmarking on target.

Command-line arguments for classes
----------------------------------

Each class (:ref:`dataset-api`, :ref:`modelwrapper-api`, :ref:`modelcompiler-api` and other) provided to the runnable scripts in scenarios can provide command-line arguments that configure the work of the object of the class.

Each class in the ``kenning.core`` implements ``form_argparse`` and ``from_argparse`` methods.
The former creates an ``argparse`` group for a given class with its parameters.
The latter takes the arguments parsed by ``argparse`` and returns the object of a class.

Model training
--------------

The ``kenning.scenarios.model_training`` performs model training using Kenning's :ref:`modelwrapper-api` and :ref:`dataset-api` objects.
To get the list of training parameters, select the model and training dataset to use (i.e. ``TensorFlowPetDatasetMobileNetV2`` model and ``PetDataset`` dataset) and run::

    python -m kenning.scenarios.model_training \
        kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
        kenning.datasets.pet_dataset.PetDataset \
        -h

This will list the possible parameters that can be used to configure the dataset, the model, and the training parameters.
For the above call, the output is as follows::

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

.. note:: The list of options depends on :ref:`modelwrapper-api` and :ref:`dataset-api`.

In the end, the training can be configured as follows::

    python -m kenning.scenarios.model_training \
        kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
        kenning.datasets.pet_dataset.PetDataset \
        --logdir build/logs \
        --dataset-root build/pet-dataset \
        --model-path build/trained-model.h5 \
        --batch-size 32 \
        --learning-rate 0.0001 \
        --num-epochs 50

This will train the model with learning rate 0.0001, batch size 32 for 50 epochs.
The trained model will be saved as ``build/trained-model.h5``.

In-framework inference performance measurements
-----------------------------------------------

The ``kenning.scenarios.inference_performance`` script runs inference os a given model in the framework it was trained on.
It requires providing:

* :ref:`modelwrapper-api`-based object wrapping the model to be tested,
* :ref:`dataset-api`-based object wrapping the dataset applicable for the model,
* a path to the output JSON file with performance and quality metrics gathered during inference by :ref:`measurements-api` object.

The example call for the method is following::

    python -m kenning.scenarios.inference_performance \
        kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
        kenning.datasets.pet_dataset.PetDataset \
        build/tensorflow_pet_dataset_mobilenetv2.json \
        --model-path kenning/resources/models/classification/tensorflow_pet_dataset_mobilenetv2.h5 \
        --dataset-root build/pet-dataset/ \
        --download-dataset

The script downloads dataset to ``build/pet-dataset`` directory, loads ``tensorflow_pet_dataset_mobilenetv2.h5`` model, runs inference on all images from the dataset and collects performance and quality metrics throughout the run.
The performance data stored in JSON file can be later rendered using :ref:`report-generation`.

ONNX conversion
---------------

The ``kenning.scenarios.onnx_conversion`` tests empirically the ONNX conversion for various frameworks and generates the report with the support matrix.
The matrix tells if model export to ONNX and model import from ONNX for a given framework and model is supported or not.
The example report with the command call is available in :doc:`onnx-conversion-support`.

The ``kenning.scenarios.onnx_conversion`` requires the list of :ref:`onnxconversion-api` classes that implement model providers and conversion method.
For the below call::

    python -m kenning.scenarios.onnx_conversion \
        build/models-directory \
        build/onnx-support.rst \
        --converters-list \
            kenning.onnxconverters.pytorch.PyTorchONNXConversion \
            kenning.onnxconverters.tensorflow.TensorFlowONNXConversion \
            kenning.onnxconverters.mxnet.MXNetONNXConversion

The conversion is tested for three frameworks - PyTorch, TensorFlow and MXNet.
The successfully converted ONNX models are stored in the ``build/models-directory``.
The final RST file with the report is stored in the ``build/onnx-support.rst`` directory.

Testing inference on target hardware
------------------------------------

The ``kenning.scenarios.inference_tester`` and ``kenning.scenarios.inference_server`` are used for testing the inference on target hardware.
The ``inference_tester`` loads the dataset and the model, compiles the model and runs inference either locally or remotely using ``inference_server``.

The ``inference_server`` receives the model, input data, and sends output data and statistics.

Both ``inference_tester`` and ``inference_server`` require :ref:`runtime-api` for determining the model execution flow.
Both scripts communicate using the communication protocol implemented in the :ref:`runtimeprotocol-api`.

In the end, the ``inference_tester`` returns the benchmark data in a form of a JSON file extracted from the ``measurements-api`` object.

The ``kenning.scenarios.inference_tester`` requires:

* :ref:`modelwrapper-api`-based class that implements model loading, I/O processing and optionally model conversion to ONNX format,
* :ref:`modelcompiler-api`-based class for compiling the model for a given target,
* :ref:`runtime-api`-based class that implements data processing and the inference method for the compiled model on the target hardware,
* :ref:`dataset-api`-based class that implements fetching of data samples and evaluation of the model,
* path to the output JSON file with performance and quality metrics.

Optionally, it requires :ref:`runtimeprotocol-api`-based class when running remotely to communicate with the ``kenning.scenarios.inference_server``.

To print the list of required arguments, run::

    python3 -m kenning.scenarios.inference_tester \                                                                                                                                                                   1468ms  Wed 28 Jul 2021 08:39:21 AM UTC
        kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
        kenning.compilers.tvm.TVMCompiler \
        kenning.runtimes.tvm.TVMRuntime \
        kenning.datasets.pet_dataset.PetDataset \
        --protocol-cls kenning.runtimeprotocols.network.NetworkProtocol \
        -h

With the above classes, the help can look as follows::

    positional arguments:                                               
      modelwrappercls       ModelWrapper-based class with inference implementation to import                                                 
      modelcompilercls      ModelCompiler-based class with compiling routines to import                                                                                                                                                                                               
      runtimecls            Runtime-based class with the implementation of model runtime                                                     
      datasetcls            Dataset-based class with dataset to import
      output                The path to the output JSON file with measurements                                                               

    optional arguments:
      -h, --help            show this help message and exit
      --protocol-cls PROTOCOL_CLS                                       
                            RuntimeProtocol-based class with the implementation of communication between inference tester and inference      
                            runner                                      
      --convert-to-onnx CONVERT_TO_ONNX
                            Before compiling the model, convert it to ONNX and use in compilation (provide a path to save here)              
      --verbosity {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                            Verbosity level

    Inference model arguments:                                          
      --model-path MODEL_PATH                                           
                            Path to the model

    Compiler arguments:                                                 
      --compiled-model-path COMPILED_MODEL_PATH
                            The path to the compiled model output
      --model-framework {onnx,keras,darknet}
                            The input type of the model, framework-wise
      --target TARGET       The kind or tag of the target device
      --target-host TARGET_HOST                                         
                            The kind or tag of the host (CPU) target device                                                                  
      --opt-level OPT_LEVEL                                             
                            The optimization level of the compilation
      --libdarknet-path LIBDARKNET_PATH
                            Path to the libdarknet.so library, for darknet models                                                            

    Runtime arguments:                                                  
      --save-model-path SAVE_MODEL_PATH
                            Path where the model will be uploaded
      --target-device-context {llvm,stackvm,cpu,c,cuda,nvptx,cl,opencl,aocl,aocl_sw_emu,sdaccel,vulkan,metal,vpi,rocm,ext_dev,hexagon,webgpu} 
                            What accelerator should be used on target device                                                                 
      --target-device-context-id TARGET_DEVICE_CONTEXT_ID
                            ID of the device to run the inference on
      --input-dtype INPUT_DTYPE                                         
                            Type of input tensor elements

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

    Runtime protocol arguments:                                         
      --host HOST           The address to the target device
      --port PORT           The port for the target device
      --packet-size PACKET_SIZE                                         
                            The maximum size of the received packets, in bytes.                                                              
      --endianness {big,little}                                         
                            The endianness of data to transfer

The ``kenning.scenarios.inference_server`` requires only:

* :ref:`runtimeprotocol-api`-based class for the implementation of the communication,
* :ref:`runtime-api`-based class for the implementation of runtime routines on device.

Both classes may require some additional arguments that can be listed with the ``-h`` flag.

The example script for ``inference_tester`` is::

    python -m kenning.scenarios.inference_tester \
        kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
        kenning.compilers.tflite.TFLiteCompiler \
        kenning.runtimes.tflite.TFLiteRuntime \
        kenning.datasets.pet_dataset.PetDataset \
        ./build/google-coral-devboard-tflite-tensorflow.json \
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

The above runs with the following ``inference_server`` setup::

    python -m kenning.scenarios.inference_server \
        kenning.runtimeprotocols.network.NetworkProtocol \
        kenning.runtimes.tflite.TFLiteRuntime \
        --host 0.0.0.0 \
        --port 12345 \
        --packet-size 32768 \
        --save-model-path /home/mendel/compiled-model.tflite \
        --delegates-list libedgetpu.so.1 \
        --verbosity INFO

.. note:: This run was tested on Google Coral Devboard device.

The ``kenning.scenarios.inference_tester`` can be also executed locally - in this case the ``--protocol-cls`` argument can be skipped.
The example call is as follows::

    python3 -m kenning.scenarios.inference_tester \
        kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \
        kenning.compilers.tvm.TVMCompiler \
        kenning.runtimes.tvm.TVMRuntime \
        kenning.datasets.pet_dataset.PetDataset \
        ./build/local-cpu-tvm-tensorflow-classification.json \
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

.. note::

     For more examples of running ``inference_tester`` and ``inference_server``, check the `kenning/scripts <https://github.com/antmicro/kenning/tree/master/scripts>`_ directory.
     In the `kenning/scripts/edge-runtimes <https://github.com/antmicro/kenning/tree/master/scripts/edge-runtimes>`_ directory there are directories with scripts for client and server calls for various target devices, deep learning frameworks and compilation frameworks.

Running inference
-----------------

The ``kenning.scenarios.inference_runner`` is used to run inference locally on a pre-compiled model.

The ``kenning.scenarios.inference_runner`` requires:

* :ref:`modelwrapper-api`-based class that performs I/O processing specific to the model,
* :ref:`runtime-api`-based class that runs inference on target using the compiled model,
* :ref:`dataprovider-api`-based class that implements fetching of data samples from various sources,
* list of :ref:`outputcollector-api`-based classes that implement output processing for the specific use-case.

To print the list of required arguments, run::

    python3 -m kenning.scenarios.inference_runner \
        kenning.modelwrappers.detectors.darknet_coco.TVMDarknetCOCOYOLOV3 \
        kenning.runtimes.tvm.TVMRuntime \
        kenning.dataproviders.camera_dataprovider.CameraDataProvider \
         --output-collectors kenning.outputcollectors.name_printer.NamePrinter \
        -h

With the above classes, the help can look as follows::

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

The example script for ``inference_runner`` is::

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

.. _report-generation:

Generating performance reports
------------------------------

The ``kenning.scenarios.inference_performance`` and ``kenning.scenarios.inference_tester`` return the JSON file as the result of benchmarks.
They contain both performance metrics data, and the quality metrics data.

The data from JSON files can be analyzed, processed and visualized by the ``kenning.scenarios.render_report`` script.
This script parses the information in JSON files and returns the RST file with the report, along with visualizations.

It requires:

* JSON file with the benchmark data,
* name of the report that will be used in the RST file and for creating Sphinx refs to figures,
* RST output file name,
* ``--root-dir`` specifying the root directory of the Sphinx documentation where the RST file will be embedded (it is used to compute relative paths),
* ``--img-dir`` specifying the path where the figures should be saved,
* ``--report-types``, which is the list describing to what types the report belong.

The example call and the resulting RST file can be observed in the :doc:`sample-report`.

As for now, the available report types are:

* ``performance`` - this is the most common report type that renders information about the overall inference performance metrics, such as inference time, CPU usage, RAM usage or GPU utilization,
* ``classification`` - this report is specific to the classification task, it renders the classification-specific quality figures and metrics, as confusion matrix, accuracy, precision, G-mean,
* ``detection`` - this report is specific to the detection task, it renders the detection-specific quality figures and metrics, as recall-precision curves, mean average precision.
