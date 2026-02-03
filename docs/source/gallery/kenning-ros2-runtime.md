# Using Kenning with ROS 2 for evaluation and deployment

This section contains tutorial of instance segmentation using Kenning and ROS 2 Nodes.

In this example [YOLACT](https://github.com/dbolya/yolact?tab=readme-ov-file) (You Only Look At CoefficienTs) model for instance segmentation will be used.
Model will be deployed on GPU using Kenning compiler [TVMCompiler](https://github.com/antmicro/kenning/blob/main/kenning/optimizers/tvm.py) - which is wrapper for [TVM deep neural network compiler](https://github.com/apache/tvm).

## Requirements

For this example you need:

* [repo tool](https://gerrit.googlesource.com/git-repo/+/refs/heads/main/README.md) to clone all necessary repositories
* [Docker](https://www.docker.com/) to use a prepared environment
* [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) to provide access to the GPU in the Docker container
* Git (to download all necessary sources)

## Evaluating the model running in ROS 2 node

The demo below will demonstrate evaluation and deployment of instance segmentation model for Lindenthal dataset.

{{uses_gpu}}

{{uses_ros2}}

Steps below assume working in a containerized environment.

### Download the demo

Create a workspace directory, where all downloaded repositories will be stored:

```bash
mkdir kenning-ros2-demo && cd kenning-ros2-demo
```

Then, download all dependencies using the `repo` tool:

```bash
# Configure git user if not configured (Docker does not have user configured)
git config --global user.email "you@example.com"
git config --global user.name "Your Name"

# Obtain all sources
repo init -u https://github.com/antmicro/ros2-vision-node-base.git -m examples/manifest.xml
repo sync -j`nproc`
```

### Prepare the Docker environment

Install necessary GPU driver libraries in the container (library drivers should match host's drivers - this can be checked with `nvidia-smi`), for example:

```bash
apt update && apt install libnvidia-gl-530 -y
```

### Compile the demo and the model

In the container, first source the ROS 2 environment:

```bash
source /opt/ros/setup.sh
```

Then install current version of Kenning:

```bash
pip install "./kenning[tensorflow,object_detection,reports,onnx,docs,tflite,tvm,onnxruntime]"
```

In addition, download necessary models:

```bash
mkdir -p models
wget -P models/ https://dl.antmicro.com/kenning/models/instance_segmentation/yolact-lindenthal.onnx
wget -P models/ https://dl.antmicro.com/kenning/models/instance_segmentation/yolact-lindenthal.onnx.json
```

To build all necessary ROS 2 nodes for the demo, run:

```bash
colcon build --base-path=src --packages-select \
    kenning_computer_vision_msgs \
    cvnode_base \
    cvnode_manager \
    --cmake-args ' -DBUILD_GUI=ON' ' -DBUILD_YOLACT=ON'
```

After this, compile the YOLACT model using TVM compiler like so:

```bash
kenning optimize --json-cfg ./src/vision_node_base/examples/config/yolact-tvm-lindenthal.json
```

### Evaluate the model

Source installed nodes:

```bash
source install/setup.sh
```

Execute instance segmentation evaluation with a following launch file:

```bash
ros2 launch cvnode_base yolact_kenning_launch.py \
    backend:=tvm \
    model_path:=./build/yolact.so \
    measurements:=tvm.json \
    report_path:=tvm/report.md
```

This will run the compiled model and collect runtime statistics from running ROS 2 application.

## Run the compiled model in full application

Once the model is compiled and confirmed to work well, we can deploy Kenning's ROS 2 node encapsulating the model in a larger ROS 2 solution.
Let's use it together with [GUI Node](https://github.com/antmicro/ros2-gui-node) and [Camera Node](https://github.com/antmicro/ros2-camera-node) to display live camera feed with instance segmentation.

GUI Node itself is a library for visualizaing data from ROS 2 topics and services.
It provides tools for manipulating Widgets and data objects, used for data visualization.
GUI itself is based upon [Dear Imgui](https://github.com/ocornut/imgui) library.

Steps are similar like in the example above but you have to allow non-network local connections to X11 so that the GUI can be started from the Docker container:

```bash test-skip
xhost +local:
```

Then run docker container with a few additional parameters:

```bash test-skip
docker run -it  \
    --device=/dev/dri:/dev/dri\
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    -v $(pwd):$(pwd) \
    -w $(pwd) \
    --gpus='all,"capabilities=compute,utility,graphics,display"' \
    -e DISPLAY="$DISPLAY" \
    -e XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" \
    --network=host \
    --ipc=host \
    ghcr.io/antmicro/ros2-gui-node:kenning-ros2-demo \
    /bin/bash
```

Source ROS 2 environment and build example:

```bash test-skip
source /opt/ros/setup.sh
colcon build --base-path=src --packages-select \
    kenning_computer_vision_msgs \
    cvnode_base \
    cvnode_manager \
    --cmake-args ' -DBUILD_GUI=ON' ' -DBUILD_YOLACT=ON'
```

Source installed nodes:

```bash test-skip
source install/setup.sh
```

And execute the example as follows:
```bash test-skip
ros2 launch cvnode_base yolact_kenning_launch.py \
    backend:=tvm \
    model_path:=./build/yolact.so \
    measurements:=tvm.json \
    report_path:=tvm/report.md
```

With this, a GUI application should appear, with:

* A live view with inferenced input data
* Instance segmentation view based on predictions from Kenning
* A widget with a list of detected objects


## Example of instance segmentation using camera and GUI Node:

In this example full YOLACT instance segmentation model is going to be used with live input from the camera.

This demo requires a camera present under `/dev/videoX` path (`X` is a camera number).

Prepare a workspace for the demo:

```bash test-skip
mkdir kenning-ros2-demo && cd kenning-ros2-demo

repo init -u https://github.com/antmicro/ros2-gui-node.git -m examples/kenning-instance-segmentation/manifest.xml
repo sync -j`nproc`
```

After this, run a Docker container with necessary environment as follows:

```bash test-skip
xhost +local:
docker run -it  \
    --device=/dev/video0:/dev/video0 \
    --device=/dev/dri:/dev/dri\
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    -v $(pwd):$(pwd) \
    -w $(pwd) \
    --gpus='all,"capabilities=compute,utility,graphics,display"' \
    -e DISPLAY="$DISPLAY" \
    -e XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" \
    --network=host \
    --ipc=host \
    ghcr.io/antmicro/ros2-gui-node:kenning-ros2-demo \
    /bin/bash
```

:::{note}
If you camera is device other than `/dev/video0`, just change the forwarded device, e.g.:

```bash test-skip
--device=/dev/video0:/dev/videoN
```

Where N is the id of the camera that should be used.
:::


Install kenning with required dependencies in the image:

```bash test-skip
pip install "./kenning[object_detection]"
```

Compile the model using TVM:

```bash test-skip
kenning optimize --json-cfg src/gui_node/examples/kenning-instance-segmentation/yolact-tvm-gpu-optimization.json
```

Build necessary nodes with:

```bash test-skip
source /opt/ros/setup.sh
colcon build --base-paths src --cmake-args -DBUILD_KENNING_YOLACT_DEMO=y
```

Source installed nodes:
```bash test-skip
source install/setup.sh
```

In the end, run:
```bash test-skip
ros2 launch gui_node kenning-instance-segmentation.py use_gui:=True
```
