# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0


from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.runtime import Runtime
from kenning.dispatcher.dep_solver import DependencySolver
from kenning.utils.class_loader import load_class


class TestDependencySolver:
    def test_init_with_multiple_block_types(self):
        solver = DependencySolver(
            {
                "dataset": ["PetDataset"],
                "model_wrapper": ["PyTorchPetDatasetMobileNetV2"],
                "runtime": ["ExecuTorchRuntime"],
            }
        )
        assert len(solver.blocks) == 3
        assert load_class("PetDataset") in solver.blocks
        assert load_class("PyTorchPetDatasetMobileNetV2") in solver.blocks
        assert load_class("ExecuTorchRuntime") in solver.blocks

    def test_init_empty_config(self):
        solver = DependencySolver({})
        assert solver.blocks == []

    def test_simple_graph(self):
        class ClientBlock:
            def __init__(self, dataset: Dataset):
                pass

        class AnotherClient:
            def __init__(self, runtime: Runtime):
                pass

        solver = DependencySolver(
            {
                "dataset": ["PetDataset"],
                "runtime": ["ExecuTorchRuntime"],
                "custom": [ClientBlock, AnotherClient],
            }
        )
        graph = solver.get_dependency_graph()
        assert graph["ClientBlock"] == ["PetDataset"]
        assert graph["AnotherClient"] == ["ExecuTorchRuntime"]
        assert graph["PetDataset"] == []
        assert graph["ExecuTorchRuntime"] == []

    def test_empty_graph(self):
        solver = DependencySolver({})
        graph = solver.get_dependency_graph()
        assert graph == {}

    def test_graph_with_no_dependencies(self):
        solver = DependencySolver(
            {
                "dataset": ["PetDataset"],
                "runtime": ["ModelRuntimeRunner"],
            }
        )
        graph = solver.get_dependency_graph()
        assert graph == {
            "PetDataset": [],
            "ModelRuntimeRunner": [],
        }

    def test_complex_graph(self):
        class RuntimeBlock:
            def __init__(self, dataset: Dataset, model: ModelWrapper):
                pass

        class OptimizerBlock:
            def __init__(self, model: ModelWrapper, runtime: Runtime):
                pass

        class ReportBlock:
            def __init__(self, dataset: Dataset, runtime: Runtime):
                pass

        solver = DependencySolver(
            {
                "dataset": ["PetDataset"],
                "model_wrapper": ["PyTorchPetDatasetMobileNetV2"],
                "runtime": ["ExecuTorchRuntime"],
                "custom": [RuntimeBlock, OptimizerBlock, ReportBlock],
            }
        )
        graph = solver.get_dependency_graph()
        assert sorted(graph["RuntimeBlock"]) == [
            "PetDataset",
            "PyTorchPetDatasetMobileNetV2",
        ]
        assert sorted(graph["OptimizerBlock"]) == [
            "ExecuTorchRuntime",
            "PyTorchPetDatasetMobileNetV2",
        ]
        assert sorted(graph["ReportBlock"]) == [
            "ExecuTorchRuntime",
            "PetDataset",
        ]
        assert graph["PetDataset"] == []
        assert graph["PyTorchPetDatasetMobileNetV2"] == ["PetDataset"]
        assert graph["ExecuTorchRuntime"] == []
