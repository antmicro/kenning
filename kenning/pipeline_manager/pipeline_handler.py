# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Tuple, Union

from kenning.pipeline_manager.core import BaseDataflowHandler, add_node  # noqa: E501
from kenning.utils.pipeline_runner import parse_json_pipeline, run_pipeline
from kenning.utils import logger


class PipelineHandler(BaseDataflowHandler):
    def __init__(self):
        nodes, io_mapping = PipelineHandler.get_nodes()
        super().__init__(nodes, io_mapping)

    def parse_json(self, json_cfg):
        return parse_json_pipeline(json_cfg)

    def run_dataflow(pipeline_tuple, output_file):
        return run_pipeline(*pipeline_tuple, output=output_file)

    def destroy_dataflow(self, *args, **kwargs):
        pass

    def create_dataflow(self, pipeline: Dict) -> Dict:
        def add_block(kenning_block: dict):
            """
            Adds dataflow node based on the `kenning_block` entry.

            Parameters
            ----------
            kenning_block : dict
                Dictionary of a block that comes from the definition
                of the pipeline.
            """
            _, kenning_name = kenning_block['type'].rsplit(".", 1)
            spec_node = self.nodes[kenning_name]
            return self.pm_graph.create_node(
                spec_node,
                [[key, value]
                 for key, value in kenning_block['parameters'].items()]
                )

        node_ids = {}

        for name in ['dataset', 'model_wrapper',
                     'runtime', 'runtime_protocol']:
            if name in pipeline:
                node_ids[name] = add_block(pipeline[name])

        node_ids['optimizer'] = []
        for optimizer in pipeline.get('optimizers', []):
            node_ids['optimizer'].append(add_block(optimizer))

        def create_if_exists(from_, to):
            if from_ in node_ids and to in node_ids:
                from_, to = node_ids[from_], node_ids[to]
                self.pm_graph.create_connection(from_, to)

        create_if_exists('dataset', 'model_wrapper')
        create_if_exists('model_wrapper', 'runtime')
        create_if_exists('runtime_protocol', 'runtime')

        if len(node_ids['optimizer']) > 0:
            previous_id = node_ids['model_wrapper']
            for opt_id in node_ids['optimizer']:
                self.pm_graph.create_connection(previous_id, opt_id)
                previous_id = opt_id
            self.pm_graph.create_connection(
                node_ids['optimizer'][-1], node_ids['runtime']
            )

        return self.pm_graph.flush_graph()

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
    def get_nodes(nodes=None, io_mapping=None):
        if nodes is None:
            nodes = {}
        if io_mapping is None:
            io_mapping = {}

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
            **io_mapping,
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
