# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module implementing the emlearn compiler.
"""

from typing import Any, Dict, List, Literal, Optional

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI, ResourceURI

DECISION_TREE_INPUT_DTYPE = "int16"
DECISION_TREE_OUTPUT_DTYPE = "int32"

# Emlearn uses codegen, and requires user to define a function for interacting
# with the model.
DECISION_TREE_FUNCTION_TEMPLATE = """
{{ src }}

void emlearn_model(const uint8_t * input, uint8_t * output) {
    ((int32_t*)output)[0] = g_emlearn_model_predict(
        (int16_t*)input,
        {{ features }}
    );
}
"""


class EmlearnCompiler(Optimizer):
    """Class implementing emlearn compiler."""

    inputtypes = ["sklearn", "any"]

    outputtypes = ["emlearn"]

    arguments_structure = {
        "compiled_model_path": {
            "description": "The path to the compiled model output",
            "type": ResourceURI,
            "required": True,
        },
    }

    def __init__(
        self,
        dataset: Optional[Dataset],
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        model_wrapper: Optional[ModelWrapper] = None,
    ):
        super().__init__(
            dataset=dataset,
            compiled_model_path=compiled_model_path,
            location=location,
            model_wrapper=model_wrapper,
        )

    def _prepare_decision_tree(
        self, clf: Any, io_spec: Optional[Dict[str, List[Dict]]] = None
    ):
        import emlearn
        import jinja2

        if self.model_wrapper.dtype != DECISION_TREE_INPUT_DTYPE:
            KLogger.warning(
                f"Decision tree trained on {self.model_wrapper.dtype}, for"
                f" models optimized with emlearn {DECISION_TREE_INPUT_DTYPE}"
                " data type is supported exclusively. Using a different data"
                " type may result in severe numerical stability issues."
            )

        features = clf.n_features_in_

        cmodel = emlearn.convert(clf, method="inline")
        src = cmodel.save("g_emlearn_model")

        env = jinja2.Environment()
        template = env.from_string(DECISION_TREE_FUNCTION_TEMPLATE)
        rendered = template.render(src=src, features=features)
        io_spec["output"][0]["dtype"] = DECISION_TREE_OUTPUT_DTYPE
        io_spec["output"] = [io_spec["output"][0]]
        return rendered

    def compile(
        self,
        input_model_path: PathOrURI,
        io_spec: Optional[Dict[str, List[Dict]]] = None,
        **kwargs: Dict,
    ) -> None:
        self.model_wrapper.prepare_model()
        with open(self.compiled_model_path, "w") as compiled_model:
            compiled_model.write(
                self._prepare_decision_tree(self.model_wrapper.model, io_spec)
            )

        self.save_io_specification(self.compiled_model_path, io_spec)

    @classmethod
    def get_framework(cls) -> str:
        return "emlearn"

    @classmethod
    def get_framework_version(cls) -> str:
        import emlearn

        return emlearn.__version__
