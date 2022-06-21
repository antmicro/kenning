"""
Wrapper for TensorFlowClustering optimizer.
"""
import tensorflow as tf
from pathlib import Path
from typing import Dict, Tuple
import tensorflow_model_optimization as tfmot

from kenning.compilers.tensorflow import TensorFlowOptimizer
from kenning.core.dataset import Dataset


def kerasconversion(modelpath: Path):
    model = tf.keras.models.load_model(modelpath, compile=False)
    return model


class TensorFlowClusteringOptimizer(TensorFlowOptimizer):
    """
    The TensorFlowClustering optimizer.
    """
    outputtypes = [
        'keras'
    ]

    inputtypes = {
        'keras': kerasconversion
    }

    arguments_structure = {
        'modelframework': {
            'argparse_name': '--model-framework',
            'description': 'The input type of the model, framework-wise',
            'default': 'keras',
            'enum': list(inputtypes.keys()),
        },
        'cluster_dense': {
            'description': 'Clusterize only dense layers',
            'type': bool,
            'default': False,
        },
        'clusters_number': {
            'description': 'Number of cluster centroids that split each layer',
            'type': int,
            'default': 10,
        },
        'preserve_sparsity': {
            'description': 'Enable sparsity preservation of the given model',
            'type': bool,
            'default': False,
        },
        'fine_tune': {
            'description': 'Fine-tune the model after clustering',
            'type': bool,
            'default': False
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
            cluster_dense: bool = False,
            clusters_number: int = 10,
            preserve_sparsity: bool = False,
            fine_tune: bool = False):
        """
        The TensorFlowClustering optimizer.

        This compiler applies clustering optimization to the model.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the model - may be used for fine-tuning
        compiled_model_path : Path
            Path where compiled model will be saved
        epochs : int
            Number of epochs used for fine-tuning
        batch_size : int
            The size of a batch used for fine-tuning
        optimizer : str
            Optimizer used during the training
        disable_from_logits
            Determines whether output of the model is normalized
        modelframework : str
            Framework of the input model, used to select a proper backend
        cluster_dense : bool
            Determines if only dense layers should be clusterized
        clusters_number : int
            Number of clusters for each weight array
        disable_sparsity_preservation : bool
            Determines whether to preserve sparsity of a given model
        fine_tune : bool
            Determines whether to fine-tune the model after clustering
        """
        self.modelframework = modelframework
        self.cluster_dense = cluster_dense
        self.clusters_number = clusters_number
        self.preserve_sparsity = preserve_sparsity
        self.fine_tune = fine_tune
        self.set_input_type(modelframework)
        super().__init__(dataset, compiled_model_path, epochs, batch_size, optimizer, disable_from_logits)  # noqa: E501

    @classmethod
    def from_argparse(cls, dataset, args):
        return cls(
            dataset,
            args.compiled_model_path,
            args.epochs,
            args.batch_size,
            args.optimizer,
            args.disable_from_logits,
            args.model_framework,
            args.cluster_dense,
            args.clusters_number,
            args.preserve_sparsity,
            args.fine_tune
        )

    def compile(
            self,
            inputmodelpath: Path,
            inputshapes: Dict[str, Tuple[int, ...]],
            dtype: str = 'float32'):
        model = self.inputtypes[self.inputtype](inputmodelpath)
        self.inputdtype = dtype

        clustering_params = {
            'number_of_clusters': self.clusters_number,
            'cluster_centroids_init':
                tfmot.clustering.keras.CentroidInitialization.KMEANS_PLUS_PLUS,
            'preserve_sparsity': self.preserve_sparsity
        }

        if self.cluster_dense:
            def apply_clustering_to_dense(layer):
                if isinstance(layer, tf.keras.layers.Dense):
                    return tfmot.clustering.keras.cluster_weights(
                        layer,
                        **clustering_params
                    )
                return layer

            clustered_model = tf.keras.models.clone_model(
                model,
                clone_function=apply_clustering_to_dense,
            )
        else:
            clustered_model = tfmot.clustering.keras.cluster_weights(
                model,
                **clustering_params
            )

        if self.fine_tune:
            clustered_model = self.train_model(clustered_model)

        optimized_model = tfmot.clustering.keras.strip_clustering(
           clustered_model
        )
        optimized_model.save(
            self.compiled_model_path,
            include_optimizer=False,
            save_format='h5'
        )
