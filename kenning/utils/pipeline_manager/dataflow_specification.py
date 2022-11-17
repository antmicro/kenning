from typing import NamedTuple

# Datasets
from kenning.datasets.coco_dataset import COCODataset2017
from kenning.datasets.imagenet_dataset import ImageNetDataset
from kenning.datasets.open_images_dataset import OpenImagesDatasetV6
from kenning.datasets.pet_dataset import PetDataset
from kenning.datasets.random_dataset import RandomizedClassificationDataset

# ModelWrappers
# classification
from kenning.modelwrappers.classification.pytorch_pet_dataset import PyTorchPetDatasetMobileNetV2  # noqa: E501
from kenning.modelwrappers.classification.tensorflow_imagenet import TensorFlowImageNet  # noqa: E501
from kenning.modelwrappers.classification.tensorflow_pet_dataset import TensorFlowPetDatasetMobileNetV2  # noqa: E501
# detectors
from kenning.modelwrappers.detectors.darknet_coco import TVMDarknetCOCOYOLOV3
from kenning.modelwrappers.detectors.yolov4 import ONNXYOLOV4
# instance segmentation
from kenning.modelwrappers.instance_segmentation.pytorch_coco import PyTorchCOCOMaskRCNN  # noqa: E501
from kenning.modelwrappers.instance_segmentation.yolact import YOLACT

# RuntimeProtocols
from kenning.runtimeprotocols.network import NetworkProtocol

# Runtimes
from kenning.runtimes.iree import IREERuntime
from kenning.runtimes.onnx import ONNXRuntime
from kenning.runtimes.tflite import TFLiteRuntime
from kenning.runtimes.tvm import TVMRuntime

# Optimizers
from kenning.compilers.iree import IREECompiler
from kenning.compilers.onnx import ONNXCompiler
from kenning.compilers.tensorflow_clustering import TensorFlowClusteringOptimizer  # noqa: E501
from kenning.compilers.tensorflow_pruning import TensorFlowPruningOptimizer
from kenning.compilers.tflite import TFLiteCompiler
from kenning.compilers.tvm import TVMCompiler


class Node(NamedTuple):
    name: str
    category: str
    type: str
    cls: object


nodes = [
    # Datasets
    Node(COCODataset2017.__name__, 'Dataset', 'dataset', COCODataset2017),
    Node(ImageNetDataset.__name__, 'Dataset', 'dataset', ImageNetDataset),
    Node(OpenImagesDatasetV6.__name__, 'Dataset', 'dataset', OpenImagesDatasetV6),  # noqa: E501
    Node(PetDataset.__name__, 'Dataset', 'dataset', PetDataset),
    Node(RandomizedClassificationDataset.__name__, 'Dataset', 'dataset', RandomizedClassificationDataset),  # noqa: E501

    # ModelWrappers
    # classification
    Node(PyTorchPetDatasetMobileNetV2.__name__, 'ModelWrapper - Classification', 'model_wrapper', PyTorchPetDatasetMobileNetV2),  # noqa: E501
    Node(TensorFlowImageNet.__name__, 'ModelWrapper - Classification', 'model_wrapper', TensorFlowImageNet),  # noqa: E501
    Node(TensorFlowPetDatasetMobileNetV2.__name__, 'ModelWrapper - Classification', 'model_wrapper', TensorFlowPetDatasetMobileNetV2),  # noqa: E501
    # detectors
    Node(TVMDarknetCOCOYOLOV3.__name__, 'ModelWrapper - Detectors', 'model_wrapper', TVMDarknetCOCOYOLOV3),  # noqa: E501
    Node(ONNXYOLOV4.__name__, 'ModelWrapper - Detectors', 'model_wrapper', ONNXYOLOV4),  # noqa: E501
    # instance segmentation
    Node(PyTorchCOCOMaskRCNN.__name__, 'ModelWrapper - Segmentation', 'model_wrapper', PyTorchCOCOMaskRCNN),  # noqa: E501
    Node(YOLACT.__name__, 'ModelWrapper - Segmentation', 'model_wrapper', YOLACT),  # noqa: E501

    # RuntimeProtocols
    Node(NetworkProtocol.__name__, 'RuntimeProtocol', 'runtime_protocol', NetworkProtocol),  # noqa: E501

    # Runtimes
    Node(IREERuntime.__name__, 'Runtime', 'runtime', IREERuntime),
    Node(ONNXRuntime.__name__, 'Runtime', 'runtime', ONNXRuntime),
    Node(TFLiteRuntime.__name__, 'Runtime', 'runtime', TFLiteRuntime),
    Node(TVMRuntime.__name__, 'Runtime', 'runtime', TVMRuntime),

    # Optimizers
    Node(IREECompiler.__name__, 'Optimizer', 'optimizer', IREECompiler),
    Node(ONNXCompiler.__name__, 'Optimizer', 'optimizer', ONNXCompiler),
    Node(TensorFlowClusteringOptimizer.__name__, 'Optimizer', 'optimizer', TensorFlowClusteringOptimizer),  # noqa: E501
    Node(TensorFlowPruningOptimizer.__name__, 'Optimizer', 'optimizer', TensorFlowPruningOptimizer),  # noqa: E501
    Node(TFLiteCompiler.__name__, 'Optimizer', 'optimizer', TFLiteCompiler),
    Node(TVMCompiler.__name__, 'Optimizer', 'optimizer', TVMCompiler)
]

io_mapping = {
    'dataset': {
        'inputs': [],
        'outputs': [
            {
                'name': 'Dataset',
                'type': 'dataset',
                'required': True
            }
        ]
    },
    'model_wrapper': {
        'inputs': [
            {
                'name': 'Dataset',
                'type': 'dataset',
                'required': True
            }
        ],
        'outputs': [
            {
                'name': 'ModelWrapper',
                'type': 'model_wrapper',
                'required': True
            },
            {
                'name': 'Model',
                'type': 'model',
                'required': True
            }
        ]
    },
    'optimizer': {
        'inputs': [
            {
                'name': 'Input model',
                'type': 'model',
                'required': True
            }
        ],
        'outputs': [
            {
                'name': 'Compiled model',
                'type': 'model',
                'required': True
            }
        ]
    },
    'runtime': {
        'inputs': [
            {
                'name': 'ModelWrapper',
                'type': 'model_wrapper',
                'required': True
            },
            {
                'name': 'Model',
                'type': 'model',
                'required': True
            },
            {
                'name': 'RuntimeProtocol',
                'type': 'runtime_protocol',
                'required': False
            }
        ],
        'outputs': []
    },
    'runtime_protocol': {
        'inputs': [],
        'outputs': [
            {
                'name': 'RuntimeProtocol',
                'type': 'runtime_protocol',
                'required': True
            }
        ]
    }
}
