# Creating applications with Kenning

The [](kenningflow-api) allows running arbitrary sequence of processing blocks that provide data, execute models using Kenning existing classes and wrappers, and process results.
It can be used to quickly create applications with Kenning after optimizing the model and using it in actual use cases.

Kenning for runtime uses both existing classes, such as [](modelwrapper-api), [](runtime-api), and also dedicated [](runner-api)-based classes.
The last family of classes are actual functional blocks used in [](kenningflow-api) that can be used for:

* Obtaining data from sources - [](dataprovider-api), for example iterating over files in the filesystem, grabbing frames from the camera or downloading data from remote source,
* Processing and delivernig the data - [](outputcollector-api), for example sending the results to the client application, visualizing model results in GUI, or storing the results in summary file,
* Running and processing various models,
* Applying other actions, as additional data analysis, preprocessing, packing and more.

[](kenningflow-api) scenario definition can be saved in JSON file and then run using {{json_flow_runner_script}} script.

## JSON structure

JSON configuration consist of list of dictionaries describing each [](runner-api)-based instance.

Example [](runner-api) specification looks as follows:

```json
{
  "type": "kenning.dataproviders.camera_dataprovider.CameraDataProvider",
  "parameters": {
    "video_file_path": "/dev/video0",
    "input_memory_layout": "NCHW",
    "input_width": 608,
    "input_height": 608
  },
  "outputs": {
    "frame": "cam_frame"
  }
},
```

Each [](runner-api)'s dictionary consists of:

* `type` - class of the [](runner-api). In example it is [CameraDataProvider](https://github.com/antmicro/kenning/blob/main/kenning/dataproviders/camera_dataprovider.py),
* `parameters` - parameters passed to class constructor. In this case we specify path to video device (`/dev/video0`), expected memory format (`NCHW`), and size of images (`608x608`)
* `inputs` - (*optional*) inputs of [](runner-api) instance. In above example above there are none,
* `outputs` - (*optional*) outputs of [](runner-api) instance. In above example it is a single output - camera frame defined as flow's variable `cam_frame`.

### Runner's IO

The specification of inputs and outputs in [](runner-api) classes is the same as described in [](model-io-metadata).

### IO compatibility

IO compatibility is checked during flow JSON parsing.

[](runner-api)'s input is considered to be compatible with associated outputs if:

* in case of `numpy.ndarray`: `dtype` and `ndim` is equal and each dimension has either the same length or input's dimension is set as `-1`, which represents any length.
  Also in input spec there could be multiple valid shapes. In that case they are placed in array, i.e. `[(1, -1, -1, 3), (1, 3, -1, -1)]`,
* in other case: `type` fields are either equal or input's `type` field is `Any`.

### IO special types

If the input or output is not a `numpy.ndarray` then its type is described by `type` field, which is a string.
In example for detection output from IO specification presented above it is a `List[DectObject]`.
This is interpreted as a list of `DectObject`'s.
The `DectObject` is a named tuple describing detection output (class names, rectangle positions, score).

### IO names and mapping

The inputs and outputs present in JSON are mappings from [](runner-api)'s local IO names to flow global variables' names, i.e. one [](runner-api) can define its outputs as `{"output_name": "data"}` and another runner can use it as its input by `{"input_name": "data"}`.
Those global variables must be unique and variable defined as input must be defined in some previous block as output to prevent cycles in flow's structure.
[](runner-api)'s IO names are specific to runner type and model (for `ModelRuntimeRunner`).

```{note}
IO names can be obtained using `get_io_specification` method.
```

### Runtime example

Now we are going to create [](kenningflow-api) presenting YOLOv4 model performance.
Let's create file `flow_scenario_detection.json` and put into it following configuration:
```json
[
  {
    "type": "kenning.dataproviders.camera_dataprovider.CameraDataProvider",
    "parameters": {
      "video_file_path": "/dev/video0",
      "input_memory_layout": "NCHW",
      "input_width": 608,
      "input_height": 608
    },
    "outputs": {
      "frame": "cam_frame"
    }
  },
  {
    "type": "kenning.runners.modelruntime_runner.ModelRuntimeRunner",
    "parameters": {
      "model_wrapper": {
        "type": "kenning.modelwrappers.detectors.yolov4.ONNXYOLOV4",
        "parameters": {
          "model_path": "./kenning/resources/models/detection/yolov4.onnx"
        }
      },
      "runtime": {
        "type": "kenning.runtimes.onnx.ONNXRuntime",
        "parameters":
        {
          "save_model_path": "./kenning/resources/models/detection/yolov4.onnx",
          "execution_providers": ["CUDAExecutionProvider"]
        }
      }
    },
    "inputs": {
      "input": "cam_frame"
    },
    "outputs": {
      "detection_output": "predictions"
    }
  },
  {
    "type": "kenning.outputcollectors.real_time_visualizers.RealTimeDetectionVisualizer",
    "parameters": {
      "viewer_width": 512,
      "viewer_height": 512,
      "input_memory_layout": "NCHW",
      "input_color_format": "BGR"
    },
    "inputs": {
      "frame": "cam_frame",
      "input": "predictions"
    }
  }
]
```

This JSON creates a [](kenningflow-api) that consists of three runners - [CameraDataProvider](https://github.com/antmicro/kenning/blob/main/kenning/dataproviders/camera_dataprovider.py), [ModelRuntimeRunner](https://github.com/antmicro/kenning/blob/main/kenning/runners/modelruntime_runner.py) and [RealTimeDetectionVisualizer](https://github.com/antmicro/kenning/blob/main/kenning/outputcollectors/real_time_visualizers.py):

* The first one captures frames from camera and pass it as `cam_frame` variable.
* Next one passes `cam_frame` to detection model (in this case YOLOv4) and returns predicted detection objects as `predictions`.
* The last one gets both outputs (`cam_frame` and `predictions`) and shows visualization of detection using DearPyGui.

## Executing KenningFlow

Let's execute KenningFlow using the above configuration.

With the config saved in the `flow_scenario_detection.json` file, run the {{json_flow_runner_script}} as follows
```bash
python -m kenning.scenarios.json_flow_runner flow_scenario_detection.json
```

This module runs KenningFlow defined in given JSON file.
With provided config it should read image from the camera and visualize output of detection model YOLOv4.

## Implemented Runners

Available implementations of [](runner-api) can be found in the [Runner documentation](runner-api).

To create custom runners check [](implementing-runner).
