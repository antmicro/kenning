# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from kenning.pipeline_manager.core import BaseDataflowHandler, GraphCreator, add_node  # noqa: E501
from kenning.utils.pipeline_runner import parse_json_pipeline, run_pipeline


class PipelineHandler(BaseDataflowHandler):
    def __init__(self):
        nodes, io_mapping = PipelineHandler.get_nodes()
        super().__init__(nodes, io_mapping, PipelineGraphCreator())

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


class PipelineGraphCreator(GraphCreator):
    def reset_graph(self):
        self.type_to_id = {}
        self.id_to_type = {}
        self.optimizer_order = {}
        self.first_optimizer = None
        self.necessary_conn = {
            ('dataset', 'model_wrapper'): False,
            ('model_wrapper', 'runtime'): False
        }

    def create_node(self, node, parameters):
        node_id = self.gen_id()
        self.nodes[node_id] = {
            'type': f'{node.cls.__module__}.{node.name}',
            'parameters': parameters
        }
        if node.type == 'optimizer':
            self.type_to_id[node.type] = self.type_to_id.get(node.type, [])
            self.type_to_id[node.type].append(node_id)
        else:
            if node.type in self.type_to_id:
                raise RuntimeError("")  # TODO
            self.type_to_id[node.type] = node_id
        self.id_to_type[node_id] = node.type
        return node_id

    def create_connection(self, from_id, to_id):
        # Registers if it's one of the necessary connections, and
        # estabilishes the order of optimizers. Due to the rigid structure
        # of the pipeline, connection between nodes don't have to be
        # directly estabilished in the graph

        from_type = self.id_to_type[from_id]
        to_type = self.id_to_type[to_id]
        if from_type != "optimizer" and to_type == "optimizer":
            self.first_optimizer = to_id
        if from_type == "optimizer" and to_type == "optimizer":
            if from_id in self.optimizer_order:
                raise RuntimeError("Nonlinear optimizer arrangment")
            self.optimizer_order[from_id] = to_id
        if from_type == "optimizer" and to_type != "optimizer":
            if from_id in self.optimizer_order:
                raise RuntimeError("Nonlinear optimizer arrangment")
            self.optimizer_order[from_id] = None

        if (from_type, to_type) in self.necessary_conn:
            self.necessary_conn[(from_type, to_type)] = True

    def flush_graph(self):
        for (from_name, to_name), exists in self.necessary_conn.items():
            if not exists:
                raise RuntimeError(f"No estabilished connection between"
                                   f"{from_name} and {to_name}")

        pipeline = {}
        types = ['model_wrapper', 'runtime', 'dataset', 'runtime_protocol']
        for type_ in types:
            if type_ in self.type_to_id:
                pipeline[type_] = self.nodes[self.type_to_id[type_]]
        optimizers = []
        opt_node = self.first_optimizer
        while opt_node is not None:
            optimizers.append(self.nodes[opt_node])
            opt_node = self.optimizer_order[opt_node]
        pipeline['optimizers'] = optimizers
        self.start_new_graph()
        return pipeline
