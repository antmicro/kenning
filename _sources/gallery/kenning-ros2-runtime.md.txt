# Using Kenning with ROS 2 for evaluation, optimization and deployment

This example demonstrates how to optimize, run and evaluate an instance segmentation model using Kenning and ROS 2 nodes.

For this task, [YOLACT](https://github.com/dbolya/yolact?tab=readme-ov-file) (You Only Look At CoefficienTs) model will be used.
The model will be deployed on a CPU or GPU using Kenning's [TVMCompiler](https://github.com/antmicro/kenning/blob/main/kenning/optimizers/tvm.py), which is a wrapper for the [TVM Deep Neural Network Compiler](https://github.com/apache/tvm).

## Dependencies

For this example you need:
1. Software:
    * [repo tool](https://gerrit.googlesource.com/git-repo/+/refs/heads/main/README.md) to clone all necessary repositories
    * [Docker](https://www.docker.com/) to use a prepared environment
    * [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) to provide access to the GPU in the Docker container (**optional**)
2. Hardware:
    * A camera for streaming frames
    * A CUDA-enabled NVIDIA GPU for inference acceleration (**optional**)

## Installation

To simplify the installation, a Docker image (**Ubuntu 24.04, Python 3.12**) containing all the dependencies required to run the environment has been prepared.
You can either pull the pre-built image (**GPU, CUDA only**) or build it from scratch yourself.
Currently, following platforms are supported:
- x86_64 / arm64 (CPU)
- x86_64 / arm64 (GPU, CUDA)
- NVIDIA Jetson

> **NOTE**
>
> See [README.md](https://github.com/antmicro/ros2-gui-node/blob/main/environments/README.md) for more information about supported platforms.
> The resulting image comes with UV ready to use inside the container (venv is activated), for example `uv pip install torch` can be used.

### Download the demo

Create a workspace directory, where all downloaded repositories will be stored:
```bash
mkdir kenning-ros2-demo && cd kenning-ros2-demo
```

Then, download all dependencies using the `repo` tool:
```bash
repo init -u https://github.com/antmicro/ros2-gui-node.git -m examples/kenning-instance-segmentation/manifest.xml
repo sync -j`nproc`
```

> **NOTE**
>
> Before executing `repo` command you may need to set up git credential by typing into terminal:
>
> ``` bash
> git config --global user.email "<e-mail address>"
> git config --global user.name "Name Surname"
> ```

It downloads the following repositories:
* [Kenning](https://github.com/antmicro/kenning) for model optimization and runtime, in the `kenning` directory
* [ROS 2 Camera node](https://github.com/antmicro/ros2-camera-node) for obtaining frames from the camera and serving its parameters as ROS 2 parameters, in the `src/camera_node` directory
* [Kenning's ROS 2 messages and services](https://github.com/antmicro/ros2-kenning-computer-vision-msgs) for computer vision, in the `src/computer_vision_msgs` directory
* [ROS 2 GUI Node](https://github.com/antmicro/ros2-gui-node), in the `src/gui_node` directory

### Prepare the Docker environment

By default, running `./build-docker.sh <platform>` does not build TVM.
Since this tutorial compiles the YOLACT model with TVM, you'll need to either install your own TVM wheel after the build (see the NOTE below), or build TVM from source by passing the `--build-tvm` flag.

::::{tabs}

:::{group-tab} CPU
```bash
./src/gui_node/environments/build-docker.sh cpu
```
:::

:::{group-tab} GPU
```bash test-skip
./src/gui_node/environments/build-docker.sh gpu
```
:::

:::{group-tab} Jetson
```bash test-skip
./src/gui_node/environments/build-docker.sh jetson
```
:::
::::

> **NOTE**
>
> Omitting `--build-tvm` is faster to build, but then TVM has to be installed manually afterwards via `uv pip install "./kenning[tvm]"` (CPU) or `uv pip install "./kenning[tvm-cuda]"` (GPU) before running the steps below.
> For more details on how to use this script and what it does, refer to: [README.md](https://github.com/antmicro/ros2-gui-node/blob/main/environments/README.md)

### Running the container

Allow non-network local connections to X11 so that the GUI can be started from the Docker container:
```bash test-skip
xhost +local:
```

The pulled or built image can be run with the following command (you need to pass the appropriate platform argument):
::::{tabs}

:::{group-tab} CPU
```bash
./src/gui_node/environments/run-docker.sh cpu
```
:::

:::{group-tab} GPU
```bash test-skip
./src/gui_node/environments/run-docker.sh gpu
```
:::

:::{group-tab} Jetson
```bash test-skip
./src/gui_node/environments/run-docker.sh jetson
```
:::
::::

> **NOTE**
>
> For more details on how to use this script and what it does, refer to: [README.md](https://github.com/antmicro/ros2-gui-node/blob/main/environments/README.md)

## Install Kenning

Install Kenning with necessary dependencies:
```bash
uv pip install "./kenning[object_detection, torch, tvm, reports]"
```

## Compiling the model

**TVM compilation** involves converting the network into an **Intermediate Representation (IR)**, where computational graph optimizations are performed, including operation fusion (combining consecutive layers into one, which reduces memory transfers).
The compiler then generates machine code optimized for a specific hardware architecture (e.g., **x86, ARM, RISC-V, CUDA**), utilizing its specific instructions (e.g., **AVX-512, NEON, or Tensor Cores**) to maximize inference performance.

In Kenning, we can do this incredibly easily.
What's more, in this particular example, we don't need to configure anything, since separate scripts have been prepared for each platform.
Simply run:
::::{tabs}

:::{group-tab} CPU
```bash
kenning optimize --cfg src/gui_node/examples/kenning-instance-segmentation/yolact-tvm-cpu-optimization.yaml
```
:::

:::{group-tab} GPU
```bash test-skip
kenning optimize --cfg src/gui_node/examples/kenning-instance-segmentation/yolact-tvm-gpu-optimization.yaml
```
:::

:::{group-tab} Jetson
```bash test-skip
kenning optimize --cfg src/gui_node/examples/kenning-instance-segmentation/yolact-tvm-gpu-optimization.yaml
```
:::
::::

## Evaluation

To evaluate the model above, we can either use a YAML configuration file, or specify the required arguments for the test scenario directly in the CLI ([Using Kenning via command-line arguments](cmd-usage)):

::::{tabs}

:::{group-tab} CPU
```bash
kenning test report --cfg src/gui_node/examples/kenning-instance-segmentation/yolact-tvm-cpu-optimization.yaml
```
:::

:::{group-tab} GPU
```bash test-skip
kenning test report --cfg src/gui_node/examples/kenning-instance-segmentation/yolact-tvm-gpu-optimization.yaml
```
:::

:::{group-tab} Jetson
```bash test-skip
kenning test report --cfg src/gui_node/examples/kenning-instance-segmentation/yolact-tvm-gpu-optimization.yaml
```
:::
::::

This command will evaluate the model on the dataset, collect performance and quality metrics into the file specified by `--measurements`, and then generate a Markdown report from them (as well as HTML).

## Running the demo

[ROS2 GUI Node](https://github.com/antmicro/ros2-gui-node) is a project created for visualizing data from [ROS 2](https://www.ros.org/).
ROS 2 itself is a robotics middleware based on a publish-subscribe (pub/sub) pattern and a node-based architecture.
Conceptually, it functions much like a microservices framework, enabling the development of efficient and modular applications for edge devices.
In this architecture, every module - from the camera, through individual AI models, to the graphical user interface (GUI) - runs as a separate, independent node.

First of all, load the `setup.sh` script for ROS 2 tools:
```bash
source /opt/ros/$ROS_DISTRO/setup.sh
```

Then, build the GUI node and the Camera node with:
```bash
colcon build --base-paths src --cmake-args -DBUILD_KENNING_YOLACT_DEMO=y -DPython3_EXECUTABLE=/opt/venv/bin/python3
```

Next, load the ROS 2 environment including the newly built packages:
```bash
source install/setup.sh
```

Finally, launch Kenning, Camera node, and GUI node using the launch file:
::::{tabs}

:::{group-tab} CPU
```bash test-skip
ros2 launch gui_node kenning-instance-segmentation-cpu.py use_gui:=true
```
:::

:::{group-tab} GPU
```bash test-skip
ros2 launch gui_node kenning-instance-segmentation.py use_gui:=true
```
:::

:::{group-tab} Jetson
```bash test-skip
ros2 launch gui_node kenning-instance-segmentation.py use_gui:=true
```
:::
::::

If you don't want to use the GUI at all, run without `use_gui:=true`:
::::{tabs}

:::{group-tab} CPU
```bash timeout=60
ros2 launch gui_node kenning-instance-segmentation-cpu.py
```
:::

:::{group-tab} GPU
```bash test-skip
ros2 launch gui_node kenning-instance-segmentation.py
```
:::

:::{group-tab} Jetson
```bash test-skip
ros2 launch gui_node kenning-instance-segmentation.py
```
:::
::::

Lastly, a GUI should appear, with:
- Direct view from Camera node
- Instance segmentation view based on predictions from  Kenning (started using `kenning flow` with `./kenning-instance-segmentation.yaml` or `kenning-instance-segmentation-cpu.yaml` if you are not using a GPU)
- A widget visualizing a list of detected objects, with a possibility to filter out not interesting classes

## Summary

In this example, we used Kenning together with ROS 2 to optimize, evaluate, and deploy an instance segmentation model.
Starting from the YOLACT model, we compiled it using TVM for CPU, GPU, and Jetson platforms, evaluated its performance and quality with `kenning test`, and finally ran a live demo streaming camera frames through Kenning and visualizing the detected instances in the ROS 2 GUI Node.
The same workflow can be easily adapted to other models and datasets supported by Kenning.
