# Transfer Learning and Optimization for Instance Segmentation Model - YOLACT

This example demonstrates a complete workflow in Kenning - from transfer learning of [YOLACT](https://github.com/dbolya/yolact?tab=readme-ov-file) (a model that detects object locations and draws their outlines/masks) for office objects, model optimization (quantization and compilation using TVM), to deployment on a GPU platform running 3 models simultaneously in real time:
- [YOLACT](https://github.com/dbolya/yolact?tab=readme-ov-file) (Instance Segmentation)
- [MMPose](https://github.com/open-mmlab/mmpose) (Human Pose Estimation)
- [DINOv2](https://github.com/facebookresearch/dinov2) (Depth Estimation)

The [ROS2 GUI Node](https://github.com/antmicro/ros2-gui-node) project will be used to demonstrate its operation.

## Dependencies

To run this scenario, you will need:
1. Hardware:
   - A camera for streaming frames
   - A CUDA-enabled NVIDIA GPU for inference acceleration
2. Software:
   - [Docker](https://docs.docker.com/engine/install/) - to use a prepared environment (for ROS2)
   - [UV](https://docs.astral.sh/uv/getting-started/installation/) - to quickly install Python dependencies and manage virtual environments
   - [repo tool](https://gerrit.googlesource.com/git-repo/+/refs/heads/main/README.md) - to clone all necessary repositories
   - [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) - to provide access to the GPU in the Docker container

{{uses_gpu}}

## Installation

For transfer learning and model optimization, we can directly clone the Kenning repository from:

```bash test-skip
git clone https://github.com/antmicro/kenning.git
```

and perform the next steps ({ref}`2 <configuration-files-section>`, {ref}`3 <transfer-learning-section>`, {ref}`4 <optimization-section>`) inside it.
However, since we ultimately want to run `ros2-gui-demo`, we will use the `repo` tool for this, which will fetch other necessary repositories and dependencies alongside Kenning (including `GUI Node`, scripts for building the ROS 2 image, etc.):

```bash test-skip
mkdir kenning-transfer-learning && cd kenning-transfer-learning
```

```bash test-skip
repo init -u https://github.com/antmicro/ros2-gui-node.git -m examples/kenning-instance-segmentation/manifest.xml
repo sync -j`nproc`
mkdir build
```

::::{note}
Before executing `repo` command you may need to set up git credential by typing into terminal:

```bash
git config --global user.email "<e-mail address>"
git config --global user.name "Name Surname"
```
::::

(configuration-files-section)=

## Configuration Files

Everything in Kenning revolves around configuration files, which store configurations for various scenarios, such as model optimization.
Let's enter the directory where Kenning is located (`kenning/`).
In the `scripts/configs/` directory, we can find the [pytorch-open-images-yolact-tvm-gpu.yaml](https://github.com/antmicro/kenning/tree/main/scripts/configs/pytorch-open-images-yolact-tvm-gpu.yaml) file, which contains the YOLACT model configuration for report generation, optimization, training, and benchmarking.
The individual sections are taken into account when invoking the relevant scenarios using the `kenning [scenario] [params]` command, e.g., `kenning train --cfg path/file.yaml`.
In this case, only `model_wrapper` and `dataset` will be taken into account:

```{literalinclude} ../scripts/configs/pytorch-open-images-yolact-tvm-gpu.yaml
:language: yaml
:start-at: model_wrapper:
:end-before: optimizers:
```

More information about this can be found here: {doc}`../json-scenarios`.
Unfortunately, there is currently no single place describing all block arguments, so you need to search for them manually in the corresponding class definition file.

(transfer-learning-section)=

## Transfer Learning

[Transfer learning](https://en.wikipedia.org/wiki/Transfer_learning) involves loading a pretrained model (weights) that was trained for task `A` and, after minor modifications, adapting it so that it can be used for task `B`.
In practice, this means usually adding or modifying the prediction head (e.g., changing the number of recognized classes), freezing the rest of the model so it isn't trained, and then training this model for a few epochs.
The next step is reducing the learning rate to avoid damaging the previously learned weights, unfreezing them, and training for a few epochs so the model adapts to the new head.

### Selecting Classes for Recognition

The YOLACT model was originally trained on the [COCO](https://cocodataset.org/#home) dataset (part of the architecture, namely `ResNet-50`, was previously trained on [ImageNet](https://www.image-net.org/7)).
It is therefore natural that we want to select objects (and a dataset) that roughly resemble those from the original training - general real-world objects, rather than, for example, computer games.

For this purpose, we will use the [OpenImages](https://storage.googleapis.com/openimages/web/index.html) dataset implemented in Kenning and select a subset of classes from it - objects that can be found in an office.

:::{note}
OpenImages dataset labels are encoded (they do not contain direct names like `Tree`), and the file with all classes and their encodings can be found here: [OpenImages Class Descriptions](https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv).
:::

These classes will be:

```text
/m/02jvh9,Mug
/m/02p0tk3,Human body
/m/0k1tl,Pen
/m/020lf,Computer mouse
/m/01m2v,Computer keyboard
/m/02522,Computer monitor
/m/050k8,Mobile phone
/m/04dr76w,Bottle
/m/02dl1y,Hat
/m/0242l,Coin
/m/080hkjn,Handbag
```

Let's save these classes to the `transfer_learning.csv` file, preferably in the Kenning root directory (`kenning/`).

### Adapting the Prediction Head

Now it's time to fine-tune YOLACT so that it recognizes new objects.
Let's open the [pytorch-open-images-yolact-tvm-gpu.yaml](https://github.com/antmicro/kenning/tree/main/scripts/configs/pytorch-open-images-yolact-tvm-gpu.yaml) file and modify the `dataset` section - let's change `classes: coco` to `classes: "transfer_learning.csv"`:

```{literalinclude} ../scripts/configs/pytorch-open-images-yolact-tvm-gpu.yaml
:language: yaml
:start-at: dataset:
:end-before: optimizers:
:emphasize-lines: 7
```

Let's change `num_epochs` to `10`.
This will train the head for 10 epochs:

```{literalinclude} ../scripts/configs/pytorch-open-images-yolact-tvm-gpu.yaml
:language: yaml
:start-at: model_wrapper:
:end-before: dataset:
:emphasize-lines: 11
```

Next, let's run the container with installed dependencies, including `ROS2` and `TVM` (for simplicity, to avoid creating a virtual environment, since we will be using this container later anyway), using:

```bash test-skip
# Building the image, this may take a while
./src/gui_node/environments/build-docker.sh gpu

# Allow non-network local connections to X11 so that the GUI can be started from the Docker container
xhost +local:

# Run the container
./src/gui_node/environments/run-docker.sh gpu
```

and let's install Kenning and the required libraries with:

```bash test-skip
uv pip install --project ./kenning -e "./kenning[object_detection,pose_estimation,onnxruntime_gpu,torch]" --group tvm-cuda
```

or when not using `ROS2`, create a virtual environment:

```bash test-skip
uv venv
source .venv/bin/activate
```

:::{note}
See the [Kenning installation](https://github.com/antmicro/kenning/blob/main/README.md#kenning-installation) section for information about the Python versions currently supported.
:::

and run:

```bash
uv pip install --project ./kenning -e "./kenning[object_detection, torch]" --group tvm-cuda
```

Then, let's run the training process for the head itself (it will be modified automatically based on the number of dataset classes):

```bash
kenning train --cfg kenning/scripts/configs/pytorch-open-images-yolact-tvm-gpu.yaml
```

Depending on the available hardware, it may take a while.
Our model will be saved to `./build/yolact_finetuned.pth`.

### Fine-tuning the Rest of the Model

The next step is to unfreeze the remaining part of the model (changing `freeze_backbone: false`), increase the number of epochs (`num_epochs: 40`), and decrease the learning rate (`learning_rate: 0.00005`).
Additionally, the recently trained model (`pretrained_weights_path: ./build/yolact_finetuned.pth`) should be used as a starting point:

```{literalinclude} ../scripts/configs/pytorch-open-images-yolact-tvm-gpu.yaml
:language: yaml
:start-at: model_wrapper:
:end-before: dataset:
:emphasize-lines: 5, 7-9, 11
```

Let's train the model again, this time in its entirety:

```bash test-skip
kenning train --cfg kenning/scripts/configs/pytorch-open-images-yolact-tvm-gpu.yaml
```

The resulting model will be located in `./build/final_yolact.pth`.

(optimization-section)=

## Optimization of the Trained Model

We can speed up our model almost 5-fold by applying int8 quantization and compiling with TVM (using appropriate flags):
- Quantization consists of reducing the precision of the model's weights and activations (e.g., from `FP32` floating-point format to 8-bit integer format `INT8`).
  This translates into faster matrix operations, lower RAM/VRAM usage, and reduced energy consumption.
  However, quantization is a lossy optimization, meaning that a slight drop in model accuracy should be expected.
- TVM compilation involves converting the network into an Intermediate Representation (IR), where computational graph optimizations are performed, including operation fusion (combining consecutive layers into one, which reduces memory transfers).
  The compiler then generates machine code optimized for a specific hardware architecture (e.g., `x86`, `ARM`, `RISC-V`, `CUDA`), utilizing its specific instructions (e.g., `AVX-512`, `NEON`, or `Tensor Cores`) to maximize inference performance.

In Kenning, we can do this incredibly easily.
Simply change `target: cuda` to `target: cuda -arch=sm_86`, where `sm_86` specifies the architecture (Compute Capability) of our graphics card - in our case, the `RTX 3090`.
You can check your GPU's architecture using the following command:

```bash test-skip
nvidia-smi --query-gpu=compute_cap --format=csv
```

:::{note}
Convert the returned value (e.g., `8.6`) into the `sm_XX` format by removing the decimal point and adding the `sm_` prefix (for `8.6`, this becomes `sm_86`).
:::

Quantization (in this example, [Post-Training Quantization](https://docs.pytorch.org/TensorRT/ts/ptq.html)) is enabled using `use_int8_precision: true`.
We apply PTQ with calibration, where `dataset_percentage: 0.001` feeds a small slice of the dataset to the compiler.
This sample is required to collect activation statistics (min/max value ranges) and calculate accurate quantization scaling factors.
The final `optimizers` section for optimization is:

```{code-block} yaml
:emphasize-lines: 10, 12, 14

optimizers:
  - type: ONNXCompiler
    parameters:
      compiled_model_path: ./build/yolact.onnx

  - type: TVMCompiler
    parameters:
      model_framework: onnx
      # This can speed things up by almost 4x.
      target: cuda -arch=sm_86
      opt_level: 3
      use_int8_precision: true
      # Small calibration slice to keep quantization fast and avoid out-of-memory error
      dataset_percentage: 0.001
      compiled_model_path: ./build/yolact_tvm_int8.tar
```

Now, let's run the optimization scenario using the command:

```bash
kenning optimize --cfg kenning/scripts/configs/pytorch-open-images-yolact-tvm-gpu.yaml
```

Our optimized model is located in `./build/yolact_tvm_int8.tar`.

## Real-Time Usage Example

[ROS2 GUI Node](https://github.com/antmicro/ros2-gui-node) is a project created for visualizing data from [ROS 2](https://www.ros.org/).
ROS 2 itself is a robotics middleware based on a publish-subscribe (pub/sub) pattern and a node-based architecture.
Conceptually, it functions much like a microservices framework, enabling the development of efficient and modular applications for edge devices.
In this architecture, every module - from the camera, through individual AI models, to the graphical user interface (GUI) - runs as a separate, independent node.

Let's replace the entire contents of the [kenning-instance-segmentation.yaml](https://github.com/antmicro/ros2-gui-node/blob/main/examples/kenning-multimodel-demo/kenning-instance-segmentation.yaml), located in the `src/gui_node/examples/kenning-multimodel-demo` directory, with the following configuration:

```{code-block} yaml
:emphasize-lines: 14, 16-17, 20, 22-23

- type: kenning.dataproviders.ros2_camera_node_data_provider.ROS2CameraNodeDataProvider
  parameters:
    topic_name: camera_frame
    output_memory_layout: NCHW
    output_width: 550
    output_height: 550
  outputs:
    frame: cam_frame
    frame_original: cam_frame_original

- type: kenning.runners.modelruntime_runner.ModelRuntimeRunner
  parameters:
    model_wrapper:
      type: kenning.modelwrappers.instance_segmentation.pytorch_yolact.YOLACTOpenImages
      parameters:
        model_path: build/final_yolact.pth
        max_detections: 100
        score_threshold: 0.2
    runtime:
      type: kenning.runtimes.tvm.TVMRuntime
      parameters:
        save_model_path: build/yolact_tvm_int8.tar
        target_device_context: cuda
  inputs:
    input: cam_frame
  outputs:
    segmentation_output: predictions

- type: kenning.outputcollectors.ros2_yolact_outputcollector.ROS2YolactOutputCollector
  parameters:
    topic_name: instance_segmentation_kenning
    input_color_format: BGR
    input_memory_layout: NCHW
  inputs:
    frame_original: cam_frame_original
    output: predictions
```

In short, we changed the runtime to TVM, updated the model wrapper to point to the appropriate class, modified file paths (including the one for the optimized model), and added a dataset section from which the model will retrieve class names and map them to the model's output (integers).

Let's build the nodes:

```bash test-skip
source /opt/ros/$ROS_DISTRO/setup.bash

colcon build --base-paths src --cmake-args -DBUILD_KENNING_MULTIMODEL_DEMO=y -DPython3_EXECUTABLE=/opt/venv/bin/python3
```

And finally, let's run the demo:

```bash test-skip
source install/setup.sh

ros2 launch gui_node kenning-multimodel-demo.py use_gui:=True
```

## Summary

In this example, we used Kenning for YOLACT transfer learning and squeezed as much performance out of it as possible.
Finally, we ran it alongside MMPose and DINOv2.
Thanks to Kenning, optimization and transfer learning are incredibly simple and convenient, without any loss in performance.
