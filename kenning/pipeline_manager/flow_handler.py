# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
import itertools
from typing import Dict
from kenning.core.flow import KenningFlow
from kenning.utils.class_loader import load_class

from kenning.pipeline_manager.core import BaseDataflowHandler, add_node, get_id
from kenning.pipeline_manager.pipeline_handler import PipelineHandler


class KenningFlowHandler(BaseDataflowHandler):
    def __init__(self):
        pipeline_nodes, pipeline_io_dict = PipelineHandler.get_nodes()

        # Nodes from PipelineHandler are used only as arguments for
        # different runners. Therefore they should have no inputs and
        # only single output, themselves, so that they can be passed
        # as runner input
        io_mapping = {
            node_type: {
                'inputs': [],
                'outputs': [
                    {
                        'name': str.capitalize(node_type.replace('_', ' ')),
                        'type': node_type,
                        'required': True
                    }
                ]
            } for node_type in pipeline_io_dict.keys()
        }

        # Everything that is not Runner
        self.primitive_modules = {
            node.name for node in pipeline_nodes.values()
        }

        nodes, io_mapping = KenningFlowHandler.get_nodes(
            pipeline_nodes, io_mapping)
        super().__init__(
            nodes,
            io_mapping,
        )

    def parse_json(self, json_cfg):
        return KenningFlow.from_json(json_cfg)

    def run_dataflow(self, kenningflow, output_file):
        return kenningflow.run()

    def destroy_dataflow(self, kenningflow):
        kenningflow.cleanup()

    def create_dataflow(self, pipeline: Dict):
        # Create runner nodes and register connections between them.
        conn_to, conn_from = defaultdict(list), {}
        for kenning_node in pipeline:
            parameters = kenning_node['parameters']
            inputs = kenning_node.get('inputs', {})
            outputs = kenning_node.get('outputs', {})

            node_options = []
            primitives = []
            for name, value in parameters.items():
                if isinstance(value, dict):
                    # Primitive should be a separate node, not an option
                    primitive_name = load_class(value['type']).__name__
                    spec_node = self.nodes[primitive_name]
                    prim_options = value['parameters']
                    prim_options = [
                        [param_name, param_value]
                        for param_name, param_value in prim_options.items()
                    ]
                    prim_id = self.pm_graph.create_node(
                        spec_node,
                        prim_options
                    )
                    primitives.append(prim_id)
                else:
                    node_options.append([name, value])

            kenning_name = load_class(kenning_node['type']).__name__
            spec_node = self.nodes[kenning_name]
            node_id = self.pm_graph.create_node(spec_node, node_options)

            for primitive_id in primitives:
                self.pm_graph.create_connection(primitive_id, node_id)

            # Register connections to be later added to respective interfaces
            for global_name in inputs.values():
                conn_to[global_name].append(node_id)
            for global_name in outputs.values():
                assert global_name not in conn_from, \
                    "Invalid Kenningflow"
                conn_from[global_name] = node_id

        # Finalize connections between all nodes
        conn_names = set(conn_to.keys())
        for conn_name in conn_names:
            from_id = conn_from[conn_name]
            for to_id in conn_to[conn_name]:
                self.pm_graph.create_connection(from_id, to_id)
        return self.pm_graph.flush_graph()

    def parse_dataflow(self, dataflow: Dict):
        runners, primitives = {}, {}
        runner_list = []
        for dn in dataflow["nodes"]:
            kenning_node = [
                node for node in self.nodes if node.name == dn['name']
            ][0]
            kenning_parameters = dict(dn['options'])

            if kenning_node.name in self.primitive_modules:
                assert len(dn['interfaces']) == 1, "Primitive module should " \
                                                   "only have single output"
                _, output_interface = dn['interfaces'][0]
                primitives[output_interface['id']] = kenning_node.type, {
                    'type': f"{kenning_node.cls.__module__}.{kenning_node.cls.__name__}",  # noqa: E501
                    'parameters': kenning_parameters
                }
            else:
                for _, connection_port in dn['interfaces']:
                    new_node = {
                        'type': f"{kenning_node.cls.__module__}.{kenning_node.cls.__name__}",  # noqa: E501
                        'parameters': kenning_parameters,
                        'inputs': {},
                        'outputs': {}
                    }
                    if new_node not in runner_list:
                        runner_list.append(new_node)
                    runners[connection_port['id']] = runner_list.index(
                        new_node)

        # Add primitives to runner parameters
        for conn in dataflow['connections']:
            _, conn_from, conn_to = conn['id'], conn['from'], conn['to']
            if conn_from in primitives:
                node_to = runners[conn_to]
                primitive_type, node_from = primitives[conn_from]
                runner_list[node_to]['parameters'][primitive_type] = node_from

        def get_runner_io(runner_node):
            runner_obj = load_class(runner_node['type'])
            return runner_obj.parse_io_specification_from_json(
                runner_node['parameters']
            )

        def is_match(arg1, arg2):
            # TODO: other cases (?)
            if 'type' in arg1 and 'type' in arg2:
                return arg1['type'] == arg2['type']
            if 'shape' in arg1 and 'shape' in arg2:
                if arg1['dtype'] != arg2['dtype']:
                    return False
                shape1, shape2 = arg1['shape'], arg2['shape']
                if len(shape1) != len(shape2):
                    return False
                for dim1, dim2 in zip(shape1, shape2):
                    if dim1 != -1 and dim2 != -1 and dim1 != dim2:
                        return False
                return True
            return False

        def find_matching_arguments(
                from_runner,
                to_runner,
                from_name,
                to_name
        ):
            output_key = "processed_output"
            if output_key not in from_runner:
                output_key = "output"
            from_arguments = from_runner[output_key]
            input_key = "processed_input"
            if input_key not in to_runner:
                input_key = "input"
            to_arguments = to_runner[input_key]
            for arg1, arg2 in itertools.product(
                    from_arguments, to_arguments):
                if is_match(arg1, arg2):
                    return arg1['name'], arg2['name']
            raise RuntimeError(f"Couldn't find matching connection between"
                               f"{from_name} and {to_name}")

        # Connect runners
        for conn in dataflow['connections']:
            conn_id, conn_from, conn_to = conn['id'], conn['from'], conn['to']
            if conn_from in runners:
                node_from, node_to = runners[conn_from], runners[conn_to]
                from_io_spec = get_runner_io(runner_list[node_from])
                to_io_spec = get_runner_io(runner_list[node_to])
                from_argname, to_argame = find_matching_arguments(
                    from_io_spec,
                    to_io_spec,
                    runner_list[node_from]['type'],
                    runner_list[node_to]['type']
                )
                if from_argname in runner_list[node_from]['outputs']:
                    # Previous runner has connected to this node and named
                    # the output globally
                    conn_id = runner_list[node_from]['outputs'][from_argname]
                else:
                    runner_list[node_from]['outputs'][from_argname] = conn_id
                runner_list[node_to]['inputs'][to_argame] = conn_id

        return True, runner_list

    @staticmethod
    def get_nodes(nodes=None, io_mapping=None):
        if nodes is None:
            nodes = {}
        if io_mapping is None:
            io_mapping = {}

        # Runners
        add_node(
            nodes,
            'kenning.dataproviders.camera_dataprovider.CameraDataProvider',
            'DataProviders',
            'data_provider'
        )
        add_node(
            nodes,
            'kenning.runners.modelruntime_runner.ModelRuntimeRunner',
            'Runner',
            'runtime_runner'
        )
        add_node(
            nodes,
            'kenning.outputcollectors.detection_visualizer.DetectionVisualizer',  # noqa: E501
            'OutputCollector',
            'output_collector'
        )
        add_node(
            nodes,
            'kenning.outputcollectors.real_time_visualizers.RealTimeDetectionVisualizer',  # noqa: E501
            'OutputCollector',
            'output_collector'
        )
        add_node(
            nodes,
            'kenning.outputcollectors.real_time_visualizers.RealTimeSegmentationVisualization',  # noqa: E501
            'OutputCollector',
            'output_collector'
        )
        add_node(
            nodes,
            'kenning.outputcollectors.real_time_visualizers.RealTimeClassificationVisualization',  # noqa: E501
            'OutputCollector',
            'output_collector'
        )

        io_mapping = {
            **io_mapping,
            'data_provider': {
                'inputs': [],
                'outputs': [
                    {
                        'name': 'Data',
                        'type': 'data_runner',
                        'required': True
                    }
                ]
            },
            'runtime_runner': {
                'inputs': [
                    {
                        'name': 'Input data',
                        'type': 'data_runner',
                        'required': True
                    },
                    {
                        'name': 'Model Wrapper',
                        'type': 'model_wrapper',
                        'required': True
                    },
                    {
                        'name': 'Runtime',
                        'type': 'runtime',
                        'required': True
                    },
                    {
                        'name': 'Calibration dataset',
                        'type': 'dataset'
                    }
                ],
                'outputs': [
                    {
                        'name': 'Model output',
                        'type': 'model_output',
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
                    },
                    {
                        'name': 'Input frames',
                        'type': 'data_runner',
                        'required': True
                    }
                ],
                'outputs': []
            }
        }

        return nodes, io_mapping
