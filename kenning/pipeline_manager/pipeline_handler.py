# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict
from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.core.runtime import Runtime
from kenning.core.runtimeprotocol import RuntimeProtocol

from kenning.pipeline_manager.core import BaseDataflowHandler, GraphCreator
from kenning.pipeline_manager.node_utils import add_node, get_category_name
from kenning.utils.class_loader import get_all_subclasses  # noqa: E501
from kenning.utils.pipeline_runner import parse_json_pipeline, run_pipeline


class PipelineHandler(BaseDataflowHandler):
    """
    Defines interpretation of graphs coming from Pipeline manager as Kenning
    optimization pipelines.
    """
    def __init__(self):
        nodes, io_mapping = PipelineHandler.get_nodes()
        super().__init__(nodes, io_mapping, PipelineGraphCreator())

    def parse_json(self, json_cfg):
        return parse_json_pipeline(json_cfg)

    def run_dataflow(self, pipeline_tuple, output_file):
        return run_pipeline(*pipeline_tuple, output=output_file)

    def destroy_dataflow(self, *args, **kwargs):
        # There is no explicit method for cleanup of Kenning objects (such as
        # runtimes, optimizers etc.), so this method doesn't need to do
        # anything
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

        base_classes = [
            ('kenning.datasets', Dataset),
            ('kenning.modelwrappers', ModelWrapper),
            ('kenning.runtimeprotocols', RuntimeProtocol),
            ('kenning.runtimes', Runtime),
            ('kenning.compilers', Optimizer)
        ]
        for base_module, base_type in base_classes:
            classes = get_all_subclasses(base_module, base_type)
            for kenning_class in classes:
                add_node(
                    nodes,
                    f"{kenning_class.__module__}.{kenning_class.__name__}",
                    get_category_name(kenning_class),
                    str.lower(base_type.__name__)
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
            'modelwrapper': {
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
            'runtimeprotocol': {
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

        return nodes, io_mapping


class PipelineGraphCreator(GraphCreator):
    """
    Creates JSON defining Kenning optimization pipeline
    """
    def reset_graph(self):
        """
        Creates graph in a standard Kenning pipeline format
        """
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
                raise RuntimeError(f"There should be only one {node.type} in "
                                   f"a pipeline")
            self.type_to_id[node.type] = node_id
        self.id_to_type[node_id] = node.type
        return node_id

    def create_connection(self, from_id, to_id):
        # Registers if it's one of the necessary connections, and
        # estabilishes the order of optimizers. Due to the rigid structure
        # of the pipeline, connection between nodes don't have to be
        # directly estabilished in the graph (there is no need to modify
        # the nodes of a graph)

        error_message = ("Nonlinear optimizer arrangement. Optimizers should "
                         "be arranged as a single linear flow running from "
                         "model wrapper to runtime")

        from_type = self.id_to_type[from_id]
        to_type = self.id_to_type[to_id]
        if from_type != "optimizer" and to_type == "optimizer":
            self.first_optimizer = to_id
        if from_type == "optimizer" and to_type == "optimizer":
            if from_id in self.optimizer_order:
                raise RuntimeError(error_message)
            self.optimizer_order[from_id] = to_id
        if from_type == "optimizer" and to_type != "optimizer":
            if from_id in self.optimizer_order:
                raise RuntimeError(error_message)
            self.optimizer_order[from_id] = None

        if (from_type, to_type) in self.necessary_conn:
            self.necessary_conn[(from_type, to_type)] = True

    def flush_graph(self):
        for (from_name, to_name), exists in self.necessary_conn.items():
            if not exists:
                raise RuntimeError(f"No estabilished connection between "
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
