# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
A module for parsing the Kenning scenarios provided via JSON or command-line.
"""

import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from pipeline_manager import specification_builder
from pipeline_manager.frontend_builder import build_prepare

from kenning.core.exceptions import VisualEditorGraphParserError
from kenning.core.model import ModelWrapper
from kenning.core.runtimebuilder import RuntimeBuilder
from kenning.pipeline_manager.core import (
    SPECIFICATION_VERSION,
    BaseDataflowHandler,
    GraphCreator,
)
from kenning.pipeline_manager.node_utils import add_node, get_category_name
from kenning.utils.class_info import generate_class_info
from kenning.utils.class_loader import (
    get_all_subclasses,
    get_base_classes_dict,
)
from kenning.utils.pipeline_runner import PipelineRunner
from kenning.utils.resource_manager import ResourceManager


class PipelineHandler(BaseDataflowHandler):
    """
    Defines interpretation of graphs coming from Pipeline manager as Kenning
    optimization pipelines.
    """

    def __init__(self, **kwargs):
        self.assets_dir = self._get_asset_dir(kwargs)
        self.spec_builder = specification_builder.SpecificationBuilder(
            spec_version=SPECIFICATION_VERSION,
            assets_dir=ResourceManager().cache_dir / self.assets_dir,
        )
        nodes, io_mapping = PipelineHandler.get_nodes(self.spec_builder)
        super().__init__(
            nodes,
            io_mapping,
            PipelineGraphCreator(),
            self.spec_builder,
            **kwargs,
        )

    def _get_asset_dir(self, kwargs):
        if not (assets_dir := getattr(self, "assets_dir", None)):
            if not (assets_dir := kwargs.pop("workspace_dir", None)):
                assets_dir = ResourceManager().cache_dir / uuid.uuid4().hex
                build_prepare(assets_dir)
        return assets_dir

    def parse_json(self, json_cfg) -> PipelineRunner:
        return PipelineRunner.from_json_cfg(json_cfg)

    def run_dataflow(
        self, pipeline_runner: PipelineRunner, output_file: Path
    ) -> int:
        return pipeline_runner.run(output=output_file, run_optimizations=False)

    def optimize_dataflow(self, pipeline_runner: PipelineRunner) -> int:
        return pipeline_runner.run(run_benchmarks=False)

    def destroy_dataflow(self, *args, **kwargs):
        # There is no explicit method for cleanup of Kenning objects (such as
        # runtimes, optimizers etc.), so this method doesn't need to do
        # anything
        pass

    def create_dataflow(self, pipeline: Dict) -> Dict[str, Union[float, Dict]]:
        def add_block(kenning_block: Dict) -> Dict:
            """
            Adds dataflow node based on the `kenning_block` entry.

            Parameters
            ----------
            kenning_block : Dict
                Dictionary of a block that comes from the definition
                of the pipeline.

            Returns
            -------
            Dict
                Pipeline Manager node definition

            Raises
            ------
            VisualEditorGraphParserError:
                Raised when provided node type is not available.
            """
            *_, kenning_name = kenning_block["type"].rsplit(".", 1)
            if kenning_name not in self.nodes:
                raise VisualEditorGraphParserError(
                    f"The node type {kenning_name} is not available in the "
                    "Visual Editor.\n\nMake sure all dependencies are "
                    "installed for this class with 'kenning info'"
                )
            spec_node = self.nodes[kenning_name]
            temp = self.pm_graph.create_node(
                spec_node,
                [
                    {"name": key, "value": value}
                    for key, value in kenning_block.get(
                        "parameters", {}
                    ).items()
                ],
            )
            return temp

        self.pm_graph.start_new_graph(graph_name="Kenning pipeline")
        node_ids = {}

        block_names = [
            "dataset",
            "model_wrapper",
            "runtime",
            "runtime_builder",
            "platform",
            "protocol",
        ]
        supported_blocks = block_names + ["optimizers"]
        for name, block in pipeline.items():
            if name not in supported_blocks:
                if name == "optimization_parameters":
                    continue
                raise VisualEditorGraphParserError(
                    f"The node type {name} is not available in the "
                    "Visual Editor."
                )
            elif name in block_names:
                # For optimization flows, this handles a list of runtimes
                if isinstance(block, List):
                    node_ids[name] = []
                    for instance in pipeline.get(name, []):
                        node_ids[name].append(add_block(instance))
                else:
                    node_ids[name] = add_block(block)

        node_ids["optimizer"] = []
        for optimizer in pipeline.get("optimizers", []):
            node_ids["optimizer"].append(add_block(optimizer))

        def create_if_exists(from_, to, unwrap=False):
            if from_ in node_ids and to in node_ids:
                from_, to = node_ids[from_], node_ids[to]
                # for optimizer connection of modelwrappers
                if len(to) > 0 and unwrap:
                    for i in to:
                        self.pm_graph.create_connection(from_, i)
                else:
                    self.pm_graph.create_connection(from_, to)

        if "runtime" not in node_ids or not isinstance(
            node_ids["runtime"], List
        ):
            create_if_exists("dataset", "model_wrapper")
            create_if_exists("model_wrapper", "runtime")
            create_if_exists("protocol", "runtime")
        else:
            create_if_exists("dataset", "model_wrapper")
            create_if_exists("model_wrapper", "optimizer", True)

        iter = 0
        if len(node_ids["optimizer"]) > 0:
            previous_id = node_ids["model_wrapper"]
            if "runtime" in node_ids and isinstance(node_ids["runtime"], List):
                for id in node_ids["runtime"]:
                    for oid in node_ids["optimizer"]:
                        bundle = enumerate(
                            zip(
                                self.pm_graph.nodes[id]["name"],
                                self.pm_graph.nodes[oid]["name"],
                            )
                        )
                        corelation = next(
                            (
                                i
                                for i, (char1, char2) in bundle
                                if char1 != char2
                            ),
                            None,
                        )
                        if corelation > 2:
                            self.pm_graph.create_connection(oid, id)
                    iter += 1
            else:
                for opt_id in node_ids["optimizer"]:
                    self.pm_graph.create_connection(previous_id, opt_id)
                    previous_id = opt_id
                if "runtime" in node_ids:
                    self.pm_graph.create_connection(
                        node_ids["optimizer"][-1], node_ids["runtime"]
                    )
        return self.pm_graph.flush_graph()

    @staticmethod
    def get_nodes(
        spec_builder, nodes=None, io_mapping=None
    ) -> Tuple[Dict, Dict]:
        if nodes is None:
            nodes = {}
        if io_mapping is None:
            io_mapping = {}

        global_base_classes = get_base_classes_dict()

        # classes that pipeline mode in pipeline manager uses
        pipeline_mode_classes = [
            "kenning.datasets",
            "kenning.modelwrappers",
            "kenning.platforms",
            "kenning.protocols",
            "kenning.runtimebuilders",
            "kenning.runtimes",
            "kenning.optimizers",
        ]

        base_classes = [
            v
            for k, v in global_base_classes.items()
            if v[0] in pipeline_mode_classes
        ]

        base_type_names = {
            base_type: str.lower(base_type.__name__)
            for _, base_type in base_classes
        }
        base_type_names[ModelWrapper] = "model_wrapper"
        base_type_names[RuntimeBuilder] = "runtime_builder"
        for base_module, base_type in base_classes:
            classes = get_all_subclasses(base_module, base_type)
            for kenning_class in classes:
                node_name = (
                    f"{kenning_class.__module__}."
                    f"{kenning_class.__name__}".split(".")[-1]
                )
                spec_builder.add_node_type(
                    name=node_name,
                    category=get_category_name(kenning_class),
                    layer=base_type_names[base_type],
                )
                if kenning_class.__doc__ is not None:
                    info = generate_class_info(
                        target=kenning_class.__module__,
                        class_name=kenning_class.__name__,
                        dependencies=False,
                    )
                    spec_builder.add_node_description(
                        name=node_name, description="".join(info)
                    )
                add_node(
                    nodes,
                    f"{kenning_class.__module__}.{kenning_class.__name__}",
                    get_category_name(kenning_class),
                    base_type_names[base_type],
                )

        io_mapping = {
            **io_mapping,
            "dataset": {
                "inputs": [],
                "outputs": [
                    {"name": "Dataset", "type": "dataset", "required": True}
                ],
            },
            "model_wrapper": {
                "inputs": [
                    {"name": "Dataset", "type": "dataset", "required": True}
                ],
                "outputs": [
                    {
                        "name": "ModelWrapper",
                        "type": "model_wrapper",
                        "required": True,
                    },
                    {"name": "Model", "type": "model", "required": True},
                ],
            },
            "optimizer": {
                "inputs": [
                    {"name": "Input model", "type": "model", "required": True}
                ],
                "outputs": [
                    {
                        "name": "Compiled model",
                        "type": "model",
                        "required": True,
                    }
                ],
            },
            "runtime": {
                "inputs": [
                    {
                        "name": "ModelWrapper",
                        "type": "model_wrapper",
                        "required": True,
                    },
                    {"name": "Model", "type": "model", "required": True},
                    {
                        "name": "Protocol",
                        "type": "protocol",
                        "required": False,
                    },
                ],
                "outputs": [],
            },
            "runtime_builder": {
                "inputs": [],
                "outputs": [],
            },
            "platform": {
                "inputs": [],
                "outputs": [],
            },
            "protocol": {
                "inputs": [],
                "outputs": [
                    {"name": "Protocol", "type": "protocol", "required": True}
                ],
            },
        }

        return nodes, io_mapping


class PipelineGraphCreator(GraphCreator):
    """
    Creates JSON defining Kenning optimization pipeline.
    """

    def reset_graph(self):
        """
        Creates graph in a standard Kenning pipeline format.
        """
        self.type_to_id = {}
        self.id_to_type = {}
        self.optimizer_order = {}
        self.first_optimizer = None
        self.necessary_conn = {("dataset", "model_wrapper"): False}

    def create_node(self, node, parameters) -> str:
        node_id = self.gen_id()
        self.nodes[node_id] = {"type": node.cls_name, "parameters": parameters}
        if node.type == "optimizer":
            self.type_to_id[node.type] = self.type_to_id.get(node.type, [])
            self.type_to_id[node.type].append(node_id)
        else:
            if node.type in self.type_to_id:
                raise RuntimeError(
                    f"There should be only one {node.type} in a pipeline"
                )
            self.type_to_id[node.type] = node_id
        self.id_to_type[node_id] = node.type
        return node_id

    def create_connection(self, from_id, to_id):
        # Registers if it's one of the necessary connections, and establishes
        # the order of optimizers. Due to the rigid structure of the pipeline,
        # connection between nodes don't have to be directly establishes in
        # the graph (there is no need to modify the nodes of a graph)

        error_message = (
            "Nonlinear optimizer arrangement. Optimizers should be arranged "
            "as a single linear flow running from model wrapper to runtime"
        )

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

    def flush_graph(self) -> Dict:
        for (from_name, to_name), exists in self.necessary_conn.items():
            if not exists:
                raise RuntimeError(
                    f"No established connection between {from_name} and "
                    f"{to_name}"
                )

        pipeline = {}
        types = [
            "model_wrapper",
            "runtime",
            "runtime_builder",
            "dataset",
            "platform",
            "protocol",
        ]
        for type_ in types:
            if type_ in self.type_to_id:
                pipeline[type_] = self.nodes[self.type_to_id[type_]]
        optimizers = []
        opt_node = self.first_optimizer
        while opt_node is not None:
            optimizers.append(self.nodes[opt_node])
            if opt_node not in self.optimizer_order:
                opt_node = None
            else:
                opt_node = self.optimizer_order[opt_node]
        pipeline["optimizers"] = optimizers
        self.start_new_graph()
        return pipeline

    def find_compatible_io(self, from_id: str, to_id: str) -> Tuple[Any, Any]:
        raise NotImplementedError
