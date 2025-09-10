# Instance Segmentation using ROS 2

This section contains tutorial of instance segmentation using Kenning and ROS 2 Nodes.

In this example [YOLACT](https://github.com/dbolya/yolact?tab=readme-ov-file) model is going to be used, it stands for "You Only Look At CoefficientTs". Model itself is a fully convolutional model for real-time instance segmentation.

Model will be deployed on GPU using Kenning compiler [TVMCompiler](https://github.com/antmicro/kenning/blob/main/kenning/optimizers/tvm.py) - which is wrapper for [TVM deep neural network compiler](https://github.com/apache/tvm).

# Requirements

For this example you need:
- A camera for streaming frames
- [repo tool](https://gerrit.googlesource.com/git-repo/+/refs/heads/main/README.md) to clone all necessary repositories
- [Docker](https://www.docker.com/) to use a prepared environment
- [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) to provide access to the GPU in the Docker container

# Quickstart

{{uses_gpu}}

This is the minimal set of steps to use ROS 2 runtime and run demo application of Instance
segmentation.

Pull the latest Docker container with prepared ROS 2 environment, container.

```bash bash test-skip
docker pull ghcr.io/antmicro/ros2-gui-node:kenning-ros2-demo
```

## Downloading the demo

Create a workspace directory, where all downloaded repositories will be stored:

```bash
mkdir kenning-ros2-demo && cd kenning-ros2-demo
```

Then, download all denpendencies using the repo tool:
```bash
git config --global user.email "you@example.com"

git config --global user.name "Your Name"

repo init -u https://github.com/antmicro/ros2-vision-node-base.git -m examples/manifest.xml

repo sync -j`nproc`
```

## Starting the Docker environment

Then, run a Docker container under the **kenning-ros2-demo** directory with:
```bash test-skip
docker run -it  \
    -v $(pwd):$(pwd) \
    -w $(pwd) \
    --gpus='all,"capabilities=compute,utility,graphics,display"' \
    ghcr.io/antmicro/ros2-gui-node:kenning-ros2-demo \
    /bin/bash
```

Install necessary GPU driver for example:
```bash
apt update && apt install libnvidia-gl-530 -y
```

Then, go to the workspace directory in the container:

## Running the example

First source the ROS 2 environment:

```bash
source /opt/ros/setup.sh
```

Then install kenning:
```bash
pip install "./kenning[tensorflow,object_detection,reports,onnx,docs,tflite,tvm,onnxruntime]"
```

Download required models:
```bash
mkdir -p models
wget -P models/ https://dl.antmicro.com/kenning/models/instance_segmentation/yolact-lindenthal.onnx
wget -P models/ https://dl.antmicro.com/kenning/models/instance_segmentation/yolact-lindenthal.onnx.json
```

Build project:
```bash
colcon build --base-path=src --packages-select \
    kenning_computer_vision_msgs \
    cvnode_base \
    cvnode_manager \
    --cmake-args ' -DBUILD_GUI=ON' ' -DBUILD_YOLACT=ON'
```

Then execute YOLACT model optimalization using Apache TVM compiler:

```bash
kenning optimize --json-cfg ./src/vision_node_base/examples/config/yolact-tvm-lindenthal.json
```

After that you are good to go to run instance segmentation demo:

Source installed nodes:
```bash
source install/setup.sh
```

Execute instance segmentation example using this launch file:
```bash
ros2 launch cvnode_base yolact_kenning_launch.py \
    backend:=tvm \
    model_path:=./build/yolact.so \
    measurements:=tvm.json \
    report_path:=tvm/report.md
```

# Run the example with GUI Node

Example provide [GUI Node](https://github.com/antmicro/ros2-gui-node) that allows you to view data gathered in the topics, like
images captured by camera node and instance segmentation output from Kenning in the real time.

GUI Node itself is a library for visualizaing data from ROS 2 topics and services. It provides tools for manipulating Widgets and data objects, used for data visualization. GUI itself
is based upon [Dear Imgui](https://github.com/ocornut/imgui) library.

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

Now execute the example again:
```bash test-skip
ros2 launch cvnode_base yolact_kenning_launch.py \
    backend:=tvm \
    model_path:=./build/yolact.so \
    measurements:=tvm.json \
    report_path:=tvm/report.md
```

GUI should appear, with:
- A live view with inferenced input data
- Instance segmentation view based on predictions from Kenning
- A widget with a list of detected objects

# Preparing the Docker environment

The Docker environment with all the necessary components is available in Dockerfile provided by example in repository [ros2-gui-node](https://github.com/antmicro/ros2-gui-node). The built image
can be pulled with:
```bash test-skip
docker pull ghcr.io/antmicro/ros2-gui-node:kenning-ros2-demo
```

or can be build.

## Building the Docker image

First of clone ros2-gui-node repository:

```bash test-skip
git clone https://github.com/antmicro/ros2-gui-node
cd ros2-gui-node
git submodule update --init --recursive
```

Then go to the directory with example:

```bash test-skip
cd examples/kenning-instance-segmentation
```

then run the command:
```bash test-skip
docker build . --tag ghcr.io/antmicro/ros2-gui-node:kenning-ros2-demo
```

Docker container itself contains:
- [ROS 2 Humble](https://docs.ros.org/en/humble/index.html) environment
- [OpenCV](https://github.com/opencv/opencv) for image processing
- [Apache TVM](https://github.com/apache/tvm) for model optimization and runtime
- CUDNN and CUDA libraries for NVIDIA GPU support
- Additional development tools

# Example using camera and GUI Node:

In this example YOLACT segmentation is going to be used with live input from the camera.

Before procedeing make sure that you have camera connected to your computer,
you can check it by running:

```bash test-skip
ls /dev/video*
```

And then you should have output, with at least one entry like:

```bash test-skip
/dev/video0 /dev/video1 ...
```

```bash test-skip
mkdir kenning-ros2-demo && cd kenning-ros2-demo
```

Then, download all denpendencies using the repo tool:
```bash test-skip
repo init -u https://github.com/antmicro/ros2-gui-node.git -m examples/kenning-instance-segmentation/manifest.xml

repo sync -j`nproc`
```


Now run docker container with additional parameter:

And don't forget about:

```bash test-skip
xhost +local:
```

```bash test-skip
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

If you camera is device other than /dev/video0 change line, to:

```bash test-skip
--device=/dev/video0:/dev/videoN
```

Where N is the id of the camera you want to use.

Install kenning with required dependencies:

```bash test-skip
pip install "./kenning[object_detection]"
```

Compile the model using TVM:

Source ROS 2 environment and build example:

```bash test-skip
source /opt/ros/setup.sh
colcon build --base-paths src --cmake-args -DBUILD_KENNING_YOLACT_DEMO=y
```

```bash test-skip
kenning optimize --json-cfg src/gui_node/examples/kenning-instance-segmentation/yolact-tvm-gpu-optimization.json
```

Source installed nodes:
```bash test-skip
source install/setup.sh
```

Now execute command:
```bash test-skip
ros2 launch gui_node kenning-instance-segmentation.py use_gui:=True
```


# Executing Kenning in ROS 2

The easiest option to execute Kenning process in ROS 2 project is to use ROS 2 launch file.

In example provided above Kenning is executed in kenning-instance-segmentation.py or
kenning-instance-segmentation-cpu.py launch file using:

```python test-skip
kenning_node = Node(
        name="kenning_node",
        executable="kenning",
        arguments=["ros","flow","--verbosity","DEBUG"],
        parameters=[{
            "config_file":"./src/gui_node/examples/kenning-instance-segmentation/kenning-instance-segmentation.json"
        }]
    )
```

It is standard Node from package launch_ros that is used to execute every ROS 2 related Node.

You can pass standard command line arguments like verbosity level using **arguments** parameters in Node. You can set different verbosity level for Kenning logger and ROS 2 logger, in example above Kenning will log DEBUG messages but DEBUG messages related to ROS 2 itself won't be showed.

If you want to see all logs, set arguments to:
```python test-skip
arguments=["--verbosity","DEBUG","--ros-args","--log-level","DEBUG"]
```

## Setting Kenning parameters

You can use parameters section of Node to set all Kenning related parameters.

To set Kenning pipeline you need to set appropriate arguments in **Node**:
```python test-skip
arguments=["ros","optimize","test" ...
```
is equivalent to running Kenning command with:
```bash test-skip
kenning optimize test ...
```

To use scenario config file, **config_file** parameter is used to provide path to the standard Kenning's scenario file:
```json test-skip
"config_file":"./src/gui_node/examples/kenning-instance-segmentation/kenning-instance-segmentation.json"
```

But you can also provide every standard command line argument supported by Kenning, using ROS 2 parameters, for example:

```python test-skip
arguments=["ros","optimize","test","--verbosity","DEBUG"],
parameters=[{
            "config_file":"./scripts/configs/tensorflow-pet-dataset-mobilenet.yml",
            "measurements":"./workspace/data.json",
            "report_path":"./report/report.md",
            "report_name":"Mobilenet Pet Dataset Test"
        }],
```

is equivalent to running the command:

```bash test-skip
kenning optimize test --cfg ./scripts/configs/tensorflow-pet-dataset-mobilenet.yml --measurements ./workspace/data.json --report-path ./report/report.md --report-name "Mobilenet Pet Dataset Test"
```
