# Defining Kenning Flow

The `KenningFlow` allows running arbitrary sequence of runners that provide data, execute model or process results.
Flow definition can be saved in JSON file and then run using {{json_flow_runner_script}} script.

## JSON structure

JSON configuration consist of list of dictionaries describing each runner. It has to determine its:

* `type` - class of the runner,
* `parameters` - parameters passed to class constructor,
* `inputs` - (optional) inputs of runner,
* `outputs` - (optional) outputs of runner.

In example, IO specification (of YOLOv4 model) looks as follows:
```{code-block} json
---
emphasize-lines: 4, 11, 16, 21, 28
---
{
  "input": [
    {
      "name": "input",
      "shape": [1, 3, 608, 608],
      "dtype": "float32"
    }
  ],
  "output": [
    {
      "name": "output",
      "shape": [1, 255, 76, 76],
      "dtype": "float32"
    },
    {
      "name": "output.3",
      "shape": [1, 255, 38, 38],
      "dtype": "float32"
    },
    {
      "name": "output.7",
      "shape": [1, 255, 19, 19],
      "dtype": "float32"
    }
  ],
  "processed_output": [
    {
      "name": "detection_output",
      "type": "List[DectObject]"
    }
  ]
}
```

We can see here several IO that can be accessed in the flow.
For example, the postprocessed output named `"detection_output"` can be used for visualization by one of the output collector runners.

### IO compatibility

IO compatibility is checked during flow JSON parsing.
Runner's input is considered to be compatible with associated outputs if:
* in case of `numpy.ndarray`: `dtype` and `ndim` is equal and each dimension has either the same length or input's dimension is set as `-1`, which represents any length.
  Also in input spec there could be multiple valid shapes. In that case they are placed in array, i.e. `[(1, -1, -1, 3), (1, 3, -1, -1)]`,
* in other case: `type` fields are either equal or input's `type` field is `Any`.

### IO special types

If the input or output is not a `numpy.ndarray` then its type is described by `type` field, which is a string.
For example for detection data it is a `List[DectObject]`.
This is interpreted as a list of `DectoObject`'s.
The `DectObject` is defined as follow:

```python
DectObject = namedtuple(
    'DectObject',
    [
        'clsname',
        'xmin',
        'ymin',
        'xmax',
        'ymax',
        'score',
        'iscrowd'
    ]
)
```

### IO names and mapping

The inputs and outputs present in JSON are mapping from runner's local IO names to flow global variables names, i.e. one runner can define its outputs as `{"output_name": "data"}` and another runner can use it as its input by `{"input_name": "data"}`.
Those global variables must be unique and variable defined as input must be defined in some previous block as output to prevent cycles in flow's structure.
Runner's IO names are specific to runner type and model (for `ModelRuntimeRunner`).

```{note}
IO names can be obtained using `get_io_specification` method.
```
### Runtime example

Now we are going to create KenningFlow presenting YOLOv4 model performance.
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

This JSON creates a KenningFlow that consists of three runners - [CameraDataProvider](https://github.com/antmicro/kenning/blob/main/kenning/dataproviders/camera_dataprovider.py), [ModelRuntimeRunner](https://github.com/antmicro/kenning/blob/main/kenning/runners/modelruntime_runner.py) and [RealTimeDetectionVisualizer](https://github.com/antmicro/kenning/blob/master/kenning/outputcollectors/real_time_visualizers.py).
The first one captures frames from camera and pass it as `cam_frame` variable.
Next one passes `cam_frame` to detection model (in this case YOLOv4) and returns predicted detection objects as `predictions`.
The last one gets both outputs (`cam_frame` and `predictions`) and shows visualization of detection using DearPyGui.

## Executing KenningFlow

Let's execute KenningFlow using the above configuration.

With the config saved in the `flow_scenario_detection.json` file, run the {{json_flow_runner_script}} as follows
```bash
python -m kenning.scenarios.json_flow_runner flow_scenario_detection.json
```

This module runs KenningFlow defined in given JSON file.
With provided config it should read image from the camera and visualize output of detection model YOLOv4.

## Implemented Runners

Runners available for use can be found here: [](runner-api).

To create custom runners please read [](implementing-runner).
