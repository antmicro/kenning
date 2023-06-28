# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for TensorFlowPruning optimizer.
"""
import tensorflow as tf
from pathlib import Path
from typing import Optional, Dict, List
import tensorflow_model_optimization as tfmot

from kenning.compilers.tensorflow_optimizers import TensorFlowOptimizer
from kenning.core.dataset import Dataset


def kerasconversion(modelpath: Path):
    model = tf.keras.models.load_model(modelpath, compile=False)
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
        'modelframework': {
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
        }
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: Path,
            epochs: int = 10,
            batch_size: int = 32,
            optimizer: str = 'adam',
            disable_from_logits: bool = False,
            modelframework: str = 'keras',
            prune_dense: bool = False,
            target_sparsity: float = 0.1):
        """
        The TensorFlowPruning optimizer.

        This compiler applies pruning optimization to the model.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the model - will be used for fine-tuning.
        compiled_model_path : Path
            Path where compiled model will be saved.
        epochs : int
            Number of epochs used for fine-tuning.
        batch_size : int
            The size of a batch used for fine-tuning.
        optimizer : str
            Optimizer used during the training
        disable_from_logits : bool
            Determines whether output of the model is normalized.
        modelframework : str
            Framework of the input model, used to select a proper backend.
        prune_dense : bool
            Determines if only dense layers should be pruned.
        target_sparsity : float
            Target weights sparsity of the model after pruning.
        """
        self.modelframework = modelframework
        self.prune_dense = prune_dense
        self.target_sparsity = target_sparsity
        self.set_input_type(modelframework)
        super().__init__(dataset, compiled_model_path, epochs, batch_size, optimizer, disable_from_logits)  # noqa: E501

    def compile(
            self,
            inputmodelpath: Path,
            io_spec: Optional[Dict[str, List[Dict]]] = None):

        model = self.inputtypes[self.inputtype](inputmodelpath)

        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=self.target_sparsity,
                begin_step=0
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

        pruned_model = self.train_model(
            pruned_model,
            [tfmot.sparsity.keras.UpdatePruningStep()]
        )

        optimized_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        optimized_model.save(
            self.compiled_model_path,
            include_optimizer=False,
            save_format='h5'
        )

        self.save_io_specification(inputmodelpath, io_spec)
