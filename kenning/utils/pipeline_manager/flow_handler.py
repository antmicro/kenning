# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

from kenning.utils.pipeline_manager.core import BaseDataflowHandler, add_node
from kenning.utils.pipeline_manager.pipeline_handler import PipelineHandler


class KenningFlowHandler(BaseDataflowHandler):
    def __init__(self):
        pipeline_nodes, _ = PipelineHandler.get_nodes()
        nodes, io_mapping = KenningFlowHandler.get_nodes(pipeline_nodes)
        super().__init__(nodes, io_mapping, None, None)  # TODO

    def create_dataflow(self, pipeline: Dict):
        pass  # TODO

    def parse_dataflow(self, dataflow: Dict):
        pass  # TODO

    @staticmethod
    def get_nodes(nodes: List = None):
        if nodes is None:
            nodes = []
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
            'data_provider': {
                'inputs': [],
                'outputs': [
                    {
                        'name': 'Data',
                        'type': 'dataset',
                        'required': True
                    }
                ]
            },
            'runtime_runner': {
                'inputs': [
                    {
                        'name': 'Input data',
                        'type': 'dataset',
                        'required': True
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
                    }
                ],
                'outputs': []
            }
        }

        return nodes, io_mapping
