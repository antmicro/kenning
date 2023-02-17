# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict
from kenning.core.flow import KenningFlow

from kenning.utils.pipeline_manager.core import BaseDataflowHandler, add_node
from kenning.utils.pipeline_manager.pipeline_handler import PipelineHandler


def json_to_kenningflow(json_cfg):
    return KenningFlow.from_json(json_cfg)


def run_kenningflow(flow):
    return flow.run()


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

        nodes, io_mapping = KenningFlowHandler.get_nodes(
            pipeline_nodes, io_mapping)
        super().__init__(
            nodes,
            io_mapping,
            json_to_kenningflow,
            run_kenningflow
        )

    def create_dataflow(self, pipeline: Dict):
        pass  # TODO

    def parse_dataflow(self, dataflow: Dict):
        pass  # TODO

    @staticmethod
    def get_nodes(nodes=None, io_mapping=None):
        if nodes is None:
            nodes = []
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
