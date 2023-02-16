# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union
from collections import defaultdict as dd

from kenning.utils.pipeline_manager.core import BaseDataflowHandler, Node, add_node  # noqa: E501
from kenning.utils.pipeline_runner import parse_json_pipeline, run_pipeline
from kenning.utils import logger


class PipelineHandler(BaseDataflowHandler):
    def __init__(self):
        nodes, io_mapping = PipelineHandler.get_nodes()
        super().__init__(nodes, io_mapping, parse_json_pipeline, run_pipeline)

    def create_dataflow(self, pipeline: Dict) -> Dict:
        dataflow_nodes = []
        dataflow = {
            'panning': {
                'x': 0,
                'y': 0,
            },
            'scaling': 1
        }

        def dict_factory() -> Dict:
            """
            Simple wrapper for a default_dict so that values can be assigned
            without creating the entries first.

            Returns
            -------
            Dict :
                Wrapped defaultdict
            """
            return dd(dict_factory)

        def default_to_regular(d: Dict) -> Dict:
            """
            Function that converts dict_factory dictionary into a normal
            python dictionary.

            Parameters
            ----------
            d : Dict
                Dictionary to be converted into a normal python dictionary.

            Returns
            -------
            Dict :
                Converted python dictionary
            """
            if isinstance(d, dd):
                d = {k: default_to_regular(v) for k, v in d.items()}
            return d

        io_mapping_to_id = dict_factory()
        id = 0

        def get_id() -> int:
            """
            Generator for new unique id values.
            Every call returns a value increased by 1.

            Returns
            -------
            int :
                New unique id
            """
            nonlocal id
            id += 1
            return id

        node_x_offset = 50
        node_width = 300
        x_pos = 0
        y_pos = 50

        def get_new_x_position() -> int:
            """
            Generator for new x node positions.
            Every call returns a value increased by `node_width` and
            `node_x_offset`.

            Returns
            -------
            int :
                Newly generated x position.
            """
            nonlocal x_pos
            x_pos += (node_width + node_x_offset)
            return x_pos

        def add_block(kenning_block: dict, kenning_block_name: str):
            """
            Adds block entry to the dataflow definition based on the
            `kenning_block` and `kenning_block_name` arguments.

            Additionaly modifies `io_mapping_to_id` dictionary that saves
            ids of inputs and outputs of every block that is later used to
            create connections between the blocks.

            Parameters
            ----------
            kenning_block : dict
                Dictionary of a block that comes from the definition
                of the pipeline.
            kenning_block_name : str
                Name of the block from the pipeline. Valid values are based
                on the `io_mapping` dictionary.
            """
            _, cls_name = kenning_block['type'].rsplit('.', 1)
            kenning_node = [
                node for node in self.nodes if node.name == cls_name
            ][0]

            new_node = {
                'name': kenning_node.name,
                'type': kenning_node.name,
                'id': get_id(),
                'options': [
                    [
                        key,
                        value
                    ]
                    for key, value in kenning_block['parameters'].items()
                ],
                'width': node_width,
                'position': {
                    'x': get_new_x_position(),
                    'y': y_pos
                },
                'state': {},
            }

            interfaces = []
            for io_name in ['inputs', 'outputs']:
                for io_object in self.io_mapping[kenning_block_name][io_name]:
                    id = get_id()
                    interfaces.append([
                        io_object['name'],
                        {
                            'value': None,
                            'isInput': True,
                            'type': io_object['type'],
                            'id': id
                        }
                    ])

                    io_to_ids = io_mapping_to_id[kenning_block_name][io_name]
                    if kenning_block_name == 'optimizer':
                        if io_to_ids[io_object['type']]:
                            io_to_ids[io_object['type']].append(id)
                        else:
                            io_to_ids[io_object['type']] = [id]
                    else:
                        io_to_ids[io_object['type']] = id

            new_node['interfaces'] = interfaces
            dataflow_nodes.append(new_node)

        # Add dataset, model_wrapper blocks
        for name in ['dataset', 'model_wrapper']:
            add_block(pipeline[name], name)

        # Add optimizer blocks
        for optimizer in pipeline['optimizers']:
            add_block(optimizer, 'optimizer')

        # Add runtime block
        add_block(pipeline['runtime'], 'runtime')

        if 'runtime_protocol' in pipeline:
            add_block(pipeline['runtime_protocol'], 'runtime_protocol')

        io_mapping_to_id = default_to_regular(io_mapping_to_id)
        # This part of code is strongly related to the definition of io_mapping
        # It should be double-checked if io_mapping changes. For now this is
        # manually set, but can be altered to use io_mapping later on
        connections = []
        connections.append({
            'id': get_id(),
            'from': io_mapping_to_id['dataset']['outputs']['dataset'],
            'to': io_mapping_to_id['model_wrapper']['inputs']['dataset'],
        })
        connections.append({
            'id': get_id(),
            'from': io_mapping_to_id['model_wrapper']['outputs']['model_wrapper'],  # noqa: E501
            'to': io_mapping_to_id['runtime']['inputs']['model_wrapper'],
        })
        # Connecting the first optimizer with model_wrapper
        connections.append({
            'id': get_id(),
            'from': io_mapping_to_id['model_wrapper']['outputs']['model'],
            'to': io_mapping_to_id['optimizer']['inputs']['model'][0],
        })

        # Connecting optimizers
        for i in range(len(pipeline['optimizers']) - 1):
            connections.append({
                'id': get_id(),
                'from': io_mapping_to_id['optimizer']['outputs']['model'][i],
                'to': io_mapping_to_id['optimizer']['inputs']['model'][i + 1],
            })

        # Connecting the last optimizer with runtime
        connections.append({
            'id': get_id(),
            'from': io_mapping_to_id['optimizer']['outputs']['model'][-1],
            'to': io_mapping_to_id['runtime']['inputs']['model'],
        })

        if 'runtime_protocol' in io_mapping_to_id:
            connections.append({
                'id': get_id(),
                'from': io_mapping_to_id['runtime_protocol']['outputs']['runtime_protocol'],  # noqa: E501
                'to': io_mapping_to_id['runtime']['inputs']['runtime_protocol'],  # noqa: E501
            })

        dataflow['connections'] = connections
        dataflow['nodes'] = dataflow_nodes
        return dataflow

    def parse_dataflow(self, dataflow: Dict) -> Tuple[bool, Union[Dict, str]]:
        log = logger.get_logger()

        def return_error(msg: str) -> Tuple[bool, str]:
            """
            Logs `msg` and returns a Tuple[bool, msg]

            Parameters
            ----------
            msg : str
                Message that is logged and that is returned as a feedback
                message.

            Returns
            -------
            Tuple[bool, str] :
                Tuple that means that parsing was unsuccessful.
            """
            log.error(msg)
            return False, msg

        def strip_node(node: Dict) -> Dict:
            """
            This method picks only `type` and `parameters` keys for the node.

            Returns
            -------
            Dict :
                Dict that contains `module` and `parameters`
                values of the input `node`.
            """
            return {
                'type': node['module'],
                'parameters': node['parameters']
            }

        def get_connected_node(
                socket_id: str,
                edges: Dict,
                nodes: Dict) -> Dict:
            """
            Get node connected to a given socket_id that is the nodes.

            Parameters
            ----------
            socket_id: str
                Socket for which node is searched
            edges: Dict
                Edges to look for the connection
            nodes: Dict
                Nodes to look for the connected node

            Returns
            -------
            Dict :
                Node that is in `nodes` dictionary, is connected to the
                `socket_id` with an edge that is in `edges` dictionary.
                None otherwise
            """
            connection = [
                edge for edge in edges
                if socket_id == edge['from'] or socket_id == edge['to']
            ]

            if not connection:
                return None

            connection = connection[0]
            corresponding_socket_id = (
                connection['to']
                if connection['from'] == socket_id else
                connection['from']
            )

            for node in nodes:
                for inp in node['inputs']:
                    if corresponding_socket_id == inp['id']:
                        return node

                for out in node['outputs']:
                    if corresponding_socket_id == out['id']:
                        return node

            # The node connected wasn't int the `nodes` argument
            return None

        dataflow_nodes = dataflow['nodes']
        dataflow_edges = dataflow['connections']

        # Creating a list of every node with its kenning path and parameters
        kenning_nodes = []
        for dn in dataflow_nodes:
            kenning_node = [
                node for node in self.nodes if node.name == dn['name']
            ][0]
            kenning_parameters = dn['options']

            parameters = {name: value for name, value in kenning_parameters}
            kenning_nodes.append({
                'module': f'{kenning_node.cls.__module__}.{kenning_node.name}',
                'type': kenning_node.type,
                'parameters': parameters,
                'inputs': [
                    interface[1]
                    for interface in dn['interfaces']
                    if interface[1]['isInput']
                ],
                'outputs': [
                    interface[1]
                    for interface in dn['interfaces']
                    if not interface[1]['isInput']
                ]
            })

        pipeline = {
            'model_wrapper': [],
            'runtime': [],
            'optimizer': [],
            'dataset': []
        }

        for node in kenning_nodes:
            pipeline[node['type']].append(node)

        # Checking cardinality of the nodes
        if len(pipeline['dataset']) != 1:
            mes = (
                'Multiple instances of dataset class'
                if len(pipeline['dataset']) > 1
                else 'No dataset class instance'
            )
            return return_error(mes)
        if len(pipeline['runtime']) != 1:
            mes = (
                'Multiple instances of runtime class'
                if len(pipeline['runtime']) > 1
                else 'No runtime class instance'
            )
            return return_error(mes)
        if len(pipeline['model_wrapper']) != 1:
            mes = (
                'Multiple instances of model_wrapper class'
                if len(pipeline['model_wrapper']) > 1
                else 'No model_wrapper class instance'
            )
            return return_error(mes)

        dataset = pipeline['dataset'][0]
        model_wrapper = pipeline['model_wrapper'][0]
        runtime = pipeline['runtime'][0]
        optimizers = pipeline['optimizer']

        # Checking required connections between found nodes
        for current_node in kenning_nodes:
            inputs = current_node['inputs']
            outputs = current_node['outputs']
            node_type = current_node['type']

            for mapping, io in zip(
                [self.io_mapping[node_type]['inputs'], self.io_mapping[node_type]['outputs']],  # noqa: E501
                [inputs, outputs]
            ):
                for connection in mapping:
                    if not connection['required']:
                        continue

                    corresponding_socket = [
                        inp for inp in io
                        if inp['type'] == connection['type']
                    ][0]

                    if get_connected_node(
                        corresponding_socket['id'],
                        dataflow_edges,
                        kenning_nodes
                    ) is None:
                        return return_error(f'There is no required connection for {node_type} class')  # noqa: E501

        # finding order of optimizers
        previous_block = model_wrapper
        next_block = None
        ordered_optimizers = []
        while True:
            out_socket_id = [
                output['id']
                for output in previous_block['outputs']
                if output['type'] == 'model'
            ][0]

            next_block = get_connected_node(out_socket_id, dataflow_edges, optimizers)  # noqa: E501

            if next_block is None:
                break

            if next_block in ordered_optimizers:
                return return_error('Cycle in the optimizer connections')

            ordered_optimizers.append(next_block)
            previous_block = next_block
            next_block = None

        if len(ordered_optimizers) != len(optimizers):
            return return_error('Cycle in the optimizer connections')

        final_scenario = {
            'model_wrapper': strip_node(model_wrapper),
            'optimizers': [strip_node(optimizer) for optimizer in ordered_optimizers],  # noqa: E501
            'runtime': strip_node(runtime),
            'dataset': strip_node(dataset)
        }

        return True, final_scenario

    @staticmethod
    def get_nodes(nodes: List[Node] = None) -> Tuple[List[Node], Dict]:
        if nodes is None:
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
            'kenning.modelwrappers.detectors.darknet_coco.TVMDarknetCOCOYOLOV3',  # noqa: E501
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
            'kenning.compilers.tensorflow_clustering.TensorFlowClusteringOptimizer',  # noqa: E501
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
            },
            'output_collector': {
                'inputs': [
                    {
                        'name': 'Model output',
                        'type': 'model_output',
                        'required': True
                    }
                ],
                'outputs': []
            }
        }

        return nodes, io_mapping
