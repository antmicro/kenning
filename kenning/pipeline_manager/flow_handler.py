# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
import itertools
from typing import Any, Dict, Iterable, List
from kenning.core.flow import KenningFlow
from kenning.utils.class_loader import load_class

from kenning.pipeline_manager.core import BaseDataflowHandler, add_node, GraphCreator  # noqa: E501
from kenning.pipeline_manager.pipeline_handler import PipelineHandler


class KenningFlowHandler(BaseDataflowHandler):
    """
    Defines the Kenningflow specification to use with Pipeline Manager
    """
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
        primitive_modules = {
            node.name for node in pipeline_nodes.values()
        }

        nodes, io_mapping = KenningFlowHandler.get_nodes(
            pipeline_nodes, io_mapping)
        super().__init__(
            nodes,
            io_mapping,
            FlowGraphCreator(primitive_modules)
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


class FlowGraphCreator(GraphCreator):
    """
    Abstraction of graph generation representing Kenningflow
    """
    def __init__(self, primitive_modules: Iterable[str]):
        """
        Creates graph in the KenningFlow format

        Parameters
        ----------
        primitive_modules : Iterable[str]
            Names of kenning types that can be used as runner
            parameter
        """
        self.primitive_modules = primitive_modules
        super().__init__()

    def reset_graph(self):
        self.primitives = []
        self.connections = []

    def create_node(self, node, parameters):
        node_id = self.gen_id()
        if node.name in self.primitive_modules:
            self.nodes[node_id] = node.type, {
                'type': f"{node.cls.__module__}.{node.cls.__name__}",
                'parameters': parameters
            }
            self.primitives.append(node_id)
        else:
            self.nodes[node_id] = {
                'type': f"{node.cls.__module__}.{node.cls.__name__}",
                'parameters': parameters,
                'inputs': {},
                'outputs': {}
            }
        return node_id

    def _get_runner_io(self, node_id: str) -> Dict[str, List[Dict]]:
        """
        Parses runner name and returns its IO specification

        Parameters
        ----------
        node_id : str
            ID of input node

        Returns
        -------
        Dict[str, List[Dict]]
            IO specification defined by the runner
        """
        runner_node = self.nodes[node_id]
        runner_obj = load_class(runner_node['type'])
        return runner_obj.parse_io_specification_from_json(
            runner_node['parameters']
        )

    def _is_match(self, arg1: Dict[str, Any], arg2: Dict[str, Any]) -> bool:
        """Checks two IO specification items whether they are
        compatible with each other (there is possible connection
        between them).

        Parameters
        ----------
        arg1, arg2 : Dict[str, Any]
            Elements of list returned by runners IO specification

        Returns
        -------
        bool
            Whether IO specification items are compatible with each other
        """
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

    def find_compatible_IO(self, from_id, to_id):
        # TODO: I'm assuming here that there is only one pair of matching
        # input-output interfaces
        from_runner_io = self._get_runner_io(from_id)
        to_runner_io = self._get_runner_io(to_id)

        output_key = "processed_output"
        if output_key not in from_runner_io:
            output_key = "output"
        from_args = from_runner_io[output_key]
        input_key = "processed_input"
        if input_key not in to_runner_io:
            input_key = "input"
        to_args = to_runner_io[input_key]
        for arg1, arg2 in itertools.product(from_args, to_args):
            if self._is_match(arg1, arg2):
                return arg1['name'], arg2['name']
        from_name = self.nodes[from_id]['type'].split(".")[-1]
        to_name = self.nodes[to_id]['type'].split(".")[-1]
        raise RuntimeError(f"Couldn't find matching connection between "
                           f"{from_name} and {to_name}")

    def create_connection(self, from_id, to_id):
        if from_id in self.primitives:
            kenning_type, node = self.nodes[from_id]
            self.nodes[to_id]['parameters'][kenning_type] = node
            del self.nodes[from_id]
        else:
            # Connections between nodes can only be created at the end,
            # when all primitives have been added to appropriate runners.
            # Here connections are only registered, they will be finished
            # when the `flush_graph` method is called.
            self.connections.append((from_id, to_id))

    def flush_graph(self):
        # Finalize connection creation
        for from_id, to_id in self.connections:
            local_from, local_to = self.find_compatible_IO(from_id, to_id)
            from_ = self.nodes[from_id]
            to_ = self.nodes[to_id]
            if local_to in to_['inputs']:
                raise RuntimeError(f"Input {local_to} has more than one "
                                   f"connection")
            if local_from in from_['outputs']:
                conn_id = from_['outputs'][local_from]
            else:
                conn_id = self.gen_id()
                from_['outputs'][local_from] = conn_id
            to_['inputs'][local_to] = conn_id

        finished_graph = list(self.nodes.values())
        self.start_new_graph()
        return finished_graph
