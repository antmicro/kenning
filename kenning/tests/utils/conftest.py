import pytest


@pytest.fixture
def mock_configuration_file_contents(tmp_path):
    dataset_file = tmp_path / 'dir/dataset.json'
    runtime_file = tmp_path / 'dir/runtime.json'
    model_file = tmp_path / 'dir/modelwrapper.json'

    invalid_runtime_file = tmp_path / 'dir/runtime-invalid.json'

    dataset_file.parent.mkdir()
    dataset_file.touch()
    runtime_file.touch()
    model_file.touch()
    invalid_runtime_file.touch()

    dataset_file.write_text('{}')
    model_file.write_text('''
        {
            "type": "kenning.modelwrappers.detectors.yolov4.ONNXYOLOV4",
            "parameters": {
                "model_path": "kenning/resources/models/detection/yolov4.onnx"
            }
        }''')
    runtime_file.write_text('''
        {"type": "kenning.runtimes.onnx.ONNXRuntime",
        "parameters": {
        "save_model_path": "kenning/resources/models/detection/yolov4.onnx"
        }
    }''')
    invalid_runtime_file.write_text('''
        {
            "type": "kenning.runtimes.onnx.ONNXRuntime",
            "parameters": {
                "invalid_arg": "kenning/resources/models/detection/yolov4.onnx"
            }
        }''')
