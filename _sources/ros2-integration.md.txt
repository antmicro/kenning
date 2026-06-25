# Integration with ROS 2

This chapter describes integration of Kenning with ROS 2 nodes and communication infrastructure.

Kenning can be used together with ROS 2 for:

* Evaluation of ROS 2 nodes and subsystems in terms of performance and quality
* Running AI models supported by Kenning and exposing topics/services for accessing them from ROS 2 nodes
* Delegating evaluation of models to remote target devices using ROS 2 communication

## Running Kenning together with ROS 2

The easiest option to execute Kenning process in ROS 2 project is to use ROS 2 launch files providing `kenning` as an executable to run, with `ros` as a subcommand:

```python test-skip
from launch_ros.actions import Node

# ...

kenning_node = Node(
        name="kenning_node",
        executable="kenning",
        arguments=["ros","flow","--verbosity","DEBUG"],
        parameters=[{
            "config_file":"./src/gui_node/examples/kenning-instance-segmentation/kenning-instance-segmentation.json"
        }]
    )
```

You can pass standard command line arguments like verbosity level using **arguments** parameters in Node.
You can set different verbosity level for Kenning logger and ROS 2 logger.

If you want to see all logs for Kenning and ROS 2, set arguments to:

```python test-skip
arguments=["--verbosity","DEBUG","--ros-args","--log-level","DEBUG"]
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

## Preparing the Docker environment for ROS 2 integration

A sample Docker environment with all the necessary components is available in Dockerfile provided by example in repository [ros2-gui-node](https://github.com/antmicro/ros2-gui-node).

The image contains:

* [ROS 2 Humble](https://docs.ros.org/en/humble/index.html) environment
* [OpenCV](https://github.com/opencv/opencv) for image processing
* [Apache TVM](https://github.com/apache/tvm) for model optimization and runtime
* CUDNN and CUDA libraries for NVIDIA GPU support
* Additional development tools

### Pulling built image

The built image can be pulled with:

```bash test-skip
docker pull ghcr.io/antmicro/ros2-gui-node:kenning-ros2-demo
```

### Building the image from scratch

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
