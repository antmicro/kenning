# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for TensorFlowPruning optimizer.
"""

from typing import Dict, List, Optional

import tensorflow as tf
from typing import Literal
import tensorflow_model_optimization as tfmot

from kenning.core.dataset import Dataset
from kenning.optimizers.tensorflow_optimizers import TensorFlowOptimizer
from kenning.utils.logger import get_logger
from kenning.utils.resource_manager import PathOrURI

LOGGER = get_logger()


def kerasconversion(model_path: PathOrURI):
    model = tf.keras.models.load_model(str(model_path), compile=False)
    return model


class TensorFlowPruningOptimizer(TensorFlowOptimizer):
    """
    The TensorFlowPruning optimizer.
    """
    inputtypes = {
        'keras': kerasconversion,
    }

    outputtypes = [
        'keras'
    ]

    arguments_structure = {
        'model_framework': {
            'argparse_name': '--model-framework',
            'description': 'The input type of the model, framework-wise',
            'default': 'keras',
            'enum': list(inputtypes.keys())
        },
        'prune_dense': {
            'description': 'Prune only dense layers',
            'type': bool,
            'default': False,
        },
        'target_sparsity': {
            'description': 'Target weights sparsity of the model after pruning',  # noqa: E501
            'type': float,
            'default': 0.1
        },
        'pruning_frequency': {
            'description': 'Defines number of steps between prunings',
            'type': int,
            'default': 100,
        },
        'pruning_end': {
            'description': 'Last steps for which model can be pruned, -1 means no end',  # noqa: E501
            'type': int,
            'default': -1,
        }
    }

    def __init__(
        self,
        dataset: Dataset,
        compiled_model_path: PathOrURI,
        location: Literal['host', 'target'] = 'host',
        epochs: int = 10,
        batch_size: int = 32,
        optimizer: str = 'adam',
        disable_from_logits: bool = False,
        save_to_zip: bool = False,
        model_framework: str = 'keras',
        prune_dense: bool = False,
        target_sparsity: float = 0.1,
        pruning_frequency: int = 100,
        pruning_end: int = -1,
    ):
        """
        The TensorFlowPruning optimizer.

        This compiler applies pruning optimization to the model.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the model - will be used for fine-tuning.
        compiled_model_path : PathOrURI
            Path or URI where compiled model will be saved.
        location : Literal['host', 'target']
            Specifies where optimization should be performed in client-server
            scenario.
        epochs : int
            Number of epochs used for fine-tuning.
        batch_size : int
            The size of a batch used for fine-tuning.
        optimizer : str
            Optimizer used during the training.
        disable_from_logits : bool
            Determines whether output of the model is normalized.
        save_to_zip : bool
            Determines whether optimized model should be saved in ZIP format.
        model_framework : str
            Framework of the input model, used to select a proper backend.
        prune_dense : bool
            Determines if only dense layers should be pruned.
        target_sparsity : float
            Target weights sparsity of the model after pruning.
        pruning_frequency : int
            Number of steps between prunings.
        pruning_end : int
            Last steps for which model can be pruned, -1 means no end.
        """
        self.model_framework = model_framework
        self.prune_dense = prune_dense
        self.target_sparsity = target_sparsity
        self.pruning_frequency = pruning_frequency
        self.pruning_end = pruning_end
        self.set_input_type(model_framework)
        super().__init__(
            dataset=dataset,
            compiled_model_path=compiled_model_path,
            location=location,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            disable_from_logits=disable_from_logits,
            save_to_zip=save_to_zip,
        )

    def compile(
            self,
            input_model_path: PathOrURI,
            io_spec: Optional[Dict[str, List[Dict]]] = None):
        model = self.inputtypes[self.inputtype](input_model_path)

        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=self.target_sparsity,
                begin_step=0, frequency=self.pruning_frequency,
                end_step=self.pruning_end,
            )
        }

        if self.prune_dense:
            def apply_pruning_to_dense(layer):
                if isinstance(layer, tf.keras.layers.Dense):
                    return tfmot.sparsity.keras.prune_low_magnitude(
                        layer,
                        **pruning_params
                    )
                return layer

            pruned_model = tf.keras.models.clone_model(
                model,
                clone_function=apply_pruning_to_dense
            )
        else:
            pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
                model,
                **pruning_params
            )

        LOGGER.info("Pruning will start after dataset is perpared")
        pruned_model = self.train_model(
            pruned_model,
            [tfmot.sparsity.keras.UpdatePruningStep()]
        )

        optimized_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        self.save_model(optimized_model)

        self.save_io_specification(input_model_path, io_spec)
