# Integration with ROS 2

This example demonstrates the integration of Kenning with ROS 2 nodes and communication infrastructure.
Kenning can be used together with ROS 2 for:
* Evaluation of ROS 2 nodes and subsystems in terms of performance and quality
* Running AI models supported by Kenning and exposing topics/services for accessing them from ROS 2 nodes
* Delegating evaluation of models to remote target devices using ROS 2 communication

## Dependencies

To run this scenario, you will need:
1. Software:
   * [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/index.html) environment
   * [OpenCV](https://github.com/opencv/opencv) for image processing
   * [Apache TVM](https://github.com/apache/tvm) for model optimization and runtime (**when built with the proper argument, see below**)
   * CUDNN and CUDA libraries for NVIDIA GPU support (**if you want to use a GPU**)
   * [Docker](https://www.docker.com/) to use a prepared environment (**optional**)
   * [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) to provide access to the GPU in the Docker container (**optional**)
2. Hardware:
   * A CUDA-enabled NVIDIA GPU for inference acceleration (**optional**)

## Installation

To simplify the installation, a Docker image (**Ubuntu 24.04, Python 3.12**) containing all the dependencies required to run the environment has been prepared.
You can either pull the pre-built image (**GPU, CUDA only**) or build it from scratch yourself.
Currently, three platforms are supported:
- x86_64 / arm64 (CPU)
- x86_64 / arm64 (GPU, CUDA)
- NVIDIA Jetson

> **NOTE**
>
> See [README.md](https://github.com/antmicro/ros2-gui-node/blob/main/environments/README.md) for more information about supported platforms.
> The resulting image comes with UV ready to use inside the container (venv is activated), for example `uv pip install torch` can be used.
> TVM is not built from source by default, use `--build-tvm` when building the image, or install a prebuilt wheel manually afterwards, see [README.md](https://github.com/antmicro/ros2-gui-node/blob/main/environments/README.md) for details.

### Pulling built image

The built image can be pulled with (**GPU, CUDA only**):

```bash test-skip
docker pull ghcr.io/antmicro/ros2-gui-node:kenning-ros2-demo-gpu
```

### Building the image from scratch

Or you can build it from scratch by first cloning the repository:

```bash test-skip
git clone https://github.com/antmicro/ros2-gui-node
cd ros2-gui-node
git submodule update --init --recursive
```

Then run the bash script for building the image:

```bash test-skip
sudo ./environments/build-docker.sh <cpu | gpu | jetson> [--build-tvm]
```

> **NOTE**
>
> For more details on how to use this script and what it does, refer to: [README.md](https://github.com/antmicro/ros2-gui-node/blob/main/environments/README.md)

## Running the container

The pulled or built image can be run with the following command (you need to pass the appropriate platform argument):

```bash test-skip
sudo ./environments/run-docker.sh [cpu | gpu | jetson]
```

> **NOTE**
>
> For more details on how to use this script and what it does, refer to: [README.md](https://github.com/antmicro/ros2-gui-node/blob/main/environments/README.md)

## Running Kenning together with ROS 2

The easiest option to execute Kenning process in ROS 2 project is to use ROS 2 launch files providing `kenning` as an executable to run, with `ros` as a subcommand:

```python test-skip
from launch_ros.actions import Node
# ...
kenning_node = Node(
    name="kenning_node",
    executable="kenning",
    arguments=["ros", "flow", "--verbosity", "DEBUG"],
    parameters=[
        {
            "config_file": "./examples/kenning-instance-segmentation/kenning-instance-segmentation.yaml"
        }
    ],
)
```

You can pass standard command line arguments like verbosity level using **arguments** parameters in Node.
You can set different verbosity level for Kenning logger and ROS 2 logger.
If you want to see all logs for Kenning and ROS 2, set arguments to:

```python test-skip
arguments = ["--verbosity", "DEBUG", "--ros-args", "--log-level", "DEBUG"]
```

## Setting Kenning parameters

You can use parameters section of Node to set all Kenning-related parameters.
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
"config_file":"./examples/kenning-instance-segmentation/kenning-instance-segmentation.yaml"
```

But you can also provide every standard command line argument supported by Kenning, using ROS 2 parameters, for example:

```python test-skip
arguments = (["ros", "optimize", "test", "--verbosity", "DEBUG"],)
parameters = (
    [
        {
            "config_file": "./scripts/configs/tensorflow-pet-dataset-mobilenet.yml",
            "measurements": "./workspace/data.json",
            "report_path": "./report/report.md",
            "report_name": "Mobilenet Pet Dataset Test",
        }
    ],
)
```

is equivalent to running the command:

```bash test-skip
kenning optimize test --cfg ./scripts/configs/tensorflow-pet-dataset-mobilenet.yml --measurements ./workspace/data.json --report-path ./report/report.md --report-name "Mobilenet Pet Dataset Test"
```

## Summary

In this example, we have prepared an environment to work with ROS 2 and Kenning.
We highly recommend reading the [README.md](https://github.com/antmicro/ros2-gui-node/blob/main/environments/README.md) before building the image.
