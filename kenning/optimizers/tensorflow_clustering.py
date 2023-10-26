# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper for TensorFlowClustering optimizer.
"""
from typing import Dict, List, Optional

import tensorflow as tf
from typing import Literal
import tensorflow_model_optimization as tfmot

from kenning.core.dataset import Dataset
from kenning.optimizers.tensorflow_optimizers import TensorFlowOptimizer
from kenning.utils.logger import KLogger
from kenning.utils.resource_manager import PathOrURI


def kerasconversion(model_path: PathOrURI):
    model = tf.keras.models.load_model(str(model_path), compile=False)
    return model


class TensorFlowClusteringOptimizer(TensorFlowOptimizer):
    """
    The TensorFlowClustering optimizer.
    """

    outputtypes = ["keras"]

    inputtypes = {"keras": kerasconversion}

    arguments_structure = {
        "model_framework": {
            "argparse_name": "--model-framework",
            "description": "The input type of the model, framework-wise",
            "default": "keras",
            "enum": list(inputtypes.keys()),
        },
        "cluster_dense": {
            "description": "Clusterize only dense layers",
            "type": bool,
            "default": False,
        },
        "clusters_number": {
            "description": "Number of cluster centroids that split each layer",
            "type": int,
            "default": 10,
        },
        "preserve_sparsity": {
            "description": "Enable sparsity preservation of the given model",
            "type": bool,
            "default": False,
        },
        "fine_tune": {
            "description": "Fine-tune the model after clustering",
            "type": bool,
            "default": False,
        },
    }

    def __init__(
        self,
        dataset: Dataset,
        compiled_model_path: PathOrURI,
        location: Literal["host", "target"] = "host",
        epochs: int = 10,
        batch_size: int = 32,
        optimizer: str = "adam",
        disable_from_logits: bool = False,
        save_to_zip: bool = False,
        model_framework: str = "keras",
        cluster_dense: bool = False,
        clusters_number: int = 10,
        preserve_sparsity: bool = False,
        fine_tune: bool = False,
    ):
        """
        The TensorFlowClustering optimizer.

        This compiler applies clustering optimization to the model.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the model - may be used for fine-tuning.
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
        cluster_dense : bool
            Determines if only dense layers should be clustered.
        clusters_number : int
            Number of clusters for each weight array.
        preserve_sparsity : bool
            Determines whether to preserve sparsity of a given model.
        fine_tune : bool
            Determines whether to fine-tune the model after clustering.
        """
        self.model_framework = model_framework
        self.cluster_dense = cluster_dense
        self.clusters_number = clusters_number
        self.preserve_sparsity = preserve_sparsity
        self.fine_tune = fine_tune
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
        io_spec: Optional[Dict[str, List[Dict]]] = None,
    ):
        model = self.inputtypes[self.inputtype](input_model_path)
        for layer in model.layers:
            layer.trainable = True

        clustering_params = {
            "number_of_clusters": self.clusters_number,
            "cluster_centroids_init": tfmot.clustering.keras.CentroidInitialization.KMEANS_PLUS_PLUS,  # noqa: E501
            "preserve_sparsity": self.preserve_sparsity,
        }

        KLogger.info("Clustering model...")
        if self.cluster_dense:

            def apply_clustering_to_dense(layer):
                if isinstance(layer, tf.keras.layers.Dense):
                    return tfmot.clustering.keras.cluster_weights(
                        layer, **clustering_params
                    )
                return layer

            clustered_model = tf.keras.models.clone_model(
                model,
                clone_function=apply_clustering_to_dense,
            )
        else:
            clustered_model = tfmot.clustering.keras.cluster_weights(
                model, **clustering_params
            )

        if self.fine_tune:
            clustered_model = self.train_model(clustered_model)

        optimized_model = tfmot.clustering.keras.strip_clustering(
            clustered_model
        )
        self.save_model(optimized_model)

        self.save_io_specification(input_model_path, io_spec)
