# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from kenning.utils.class_loader import load_class
from kenning.utils.logger import get_logger
from typing import NamedTuple, List

_LOGGER = get_logger()


class Node(NamedTuple):
    """
    NamedTuple specifying the single node in specification.

    Attributes
    ----------
    name : str
        Name of the block
    category : str
        Category of the node (where it appears in the available blocks menu)
    type : str
        Type of the block (for internal usage in pipeline manager)
    cls : object
        Class object to extract parameters from.
    """
    name: str
    category: str
    type: str
    cls: object


def add_node(
        node_list: List,
        nodemodule: str,
        category: str,
        type: str):
    """
    Loads a class containing Kenning block and adds it to available nodes.

    If the class can't be imported due to import errors, it is not added.

    Parameters
    ----------
    node_list: List[Node]
        List of nodes to add to the specification
    nodemodule : str
        Python-like path to the class holding a block to add to specification
    category : str
        Category of the block
    type : str
        Type of the block added to the specification
    """
    try:
        nodeclass = load_class(nodemodule)
        node_list.append(Node(nodeclass.__name__, category, type, nodeclass))
    except (ModuleNotFoundError, ImportError, Exception) as err:
        msg = f'Could not add {nodemodule}. Reason:'
        _LOGGER.warn('-' * len(msg))
        _LOGGER.warn(msg)
        _LOGGER.warn(err)
        _LOGGER.warn('-' * len(msg))


nodes = []

# Datasets
add_node(
    nodes,
    'kenning.datasets.coco_dataset.COCODataset2017',
    'Dataset',
    'dataset'
)
add_node(
    nodes,
    'kenning.datasets.imagenet_dataset.ImageNetDataset',
    'Dataset',
    'dataset'
)
add_node(
    nodes,
    'kenning.datasets.open_images_dataset.OpenImagesDatasetV6',
    'Dataset',
    'dataset'
)
add_node(
    nodes,
    'kenning.datasets.pet_dataset.PetDataset',
    'Dataset',
    'dataset'
)
add_node(
    nodes,
    'kenning.datasets.random_dataset.RandomizedClassificationDataset',
    'Dataset',
    'dataset'
)

# ModelWrappers
# classification
add_node(
    nodes,
    'kenning.modelwrappers.classification.pytorch_pet_dataset.PyTorchPetDatasetMobileNetV2',  # noqa: E501
    'ModelWrapper - Classification',
    'model_wrapper'
)
add_node(
    nodes,
    'kenning.modelwrappers.classification.tensorflow_imagenet.TensorFlowImageNet',  # noqa: E501
    'ModelWrapper - Classification',
    'model_wrapper'
)
add_node(
    nodes,
    'kenning.modelwrappers.classification.tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2',  # noqa: E501
    'ModelWrapper - Classification',
    'model_wrapper'
)
# detectors
add_node(
    nodes,
    'kenning.modelwrappers.detectors.darknet_coco.TVMDarknetCOCOYOLOV3',
    'ModelWrapper - Detectors',
    'model_wrapper'
)
add_node(
    nodes,
    'kenning.modelwrappers.detectors.yolov4.ONNXYOLOV4',
    'ModelWrapper - Detectors',
    'model_wrapper'
)
# instance segmentation
add_node(
    nodes,
    'kenning.modelwrappers.instance_segmentation.pytorch_coco.PyTorchCOCOMaskRCNN',  # noqa: E501
    'ModelWrapper - Segmentation',
    'model_wrapper'
)
add_node(
    nodes,
    'kenning.modelwrappers.instance_segmentation.yolact.YOLACT',
    'ModelWrapper - Segmentation',
    'model_wrapper'
)

# RuntimeProtocols
add_node(
    nodes,
    'kenning.runtimeprotocols.network.NetworkProtocol',
    'RuntimeProtocol',
    'runtime_protocol'
)

# Runtimes
add_node(
    nodes,
    'kenning.runtimes.iree.IREERuntime',
    'Runtime',
    'runtime'
)
add_node(
    nodes,
    'kenning.runtimes.onnx.ONNXRuntime',
    'Runtime',
    'runtime'
)
add_node(
    nodes,
    'kenning.runtimes.tflite.TFLiteRuntime',
    'Runtime',
    'runtime'
)
add_node(
    nodes,
    'kenning.runtimes.tvm.TVMRuntime',
    'Runtime',
    'runtime'
)

# Optimizers
add_node(
    nodes,
    'kenning.compilers.iree.IREECompiler',
    'Optimizer',
    'optimizer'
)
add_node(
    nodes,
    'kenning.compilers.onnx.ONNXCompiler',
    'Optimizer',
    'optimizer'
)
add_node(
    nodes,
    'kenning.compilers.tensorflow_clustering.TensorFlowClusteringOptimizer',
    'Optimizer',
    'optimizer'
)
add_node(
    nodes,
    'kenning.compilers.tensorflow_pruning.TensorFlowPruningOptimizer',
    'Optimizer',
    'optimizer'
)
add_node(
    nodes,
    'kenning.compilers.tflite.TFLiteCompiler',
    'Optimizer',
    'optimizer'
)
add_node(
    nodes,
    'kenning.compilers.tvm.TVMCompiler',
    'Optimizer',
    'optimizer'
)

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
