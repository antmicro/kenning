"""
Wrapper for TensorFlowClustering optimizer.
"""
import tensorflow as tf
from pathlib import Path
from typing import Dict, Tuple
import tensorflow_model_optimization as tfmot

from kenning.core.optimizer import Optimizer
from kenning.core.dataset import Dataset


def kerasconversion(modelpath: Path):
    model = tf.keras.models.load_model(modelpath, compile=False)
    return model


class TensorFlowClusteringOptimizer(Optimizer):
    """
    The TensorFlowClustering optimizer.
    """
    outputtypes = [
        'keras'
    ]

    inputtypes = {
        'keras': kerasconversion
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: Path,
            modelframework: str,
            cluster_dense: bool,
            clusters_number: int,
            disable_sparsity_preservation: bool):
        """
        The TensorFlowClustering optimizer.

        This compiler applies clustering optimization to the model.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the model - may be used for quantization
            during compilation stage
        compiled_model_path : Path
            Path where compiled model will be saved
        modelframework : str
            Framework of the input model, used to select a proper backend
        cluster_dense : bool
            Determines if only dense layers should be clusterized
        clusters_number : int
            Number of clusters for each weight array
        disable_sparsity_preservation : bool
            Determines whether to preserve sparsity of a given model
        """
        self.modelframework = modelframework
        self.cluster_dense = cluster_dense
        self.clusters_number = clusters_number
        self.disable_sparsity_preservation = disable_sparsity_preservation
        self.set_input_type(modelframework)
        super().__init__(dataset, compiled_model_path)

    @classmethod
    def form_argparse(cls):
        parser, group = super().form_argparse()
        group.add_argument(
            '--model-framework',
            help='The input type of the model, framework-wise',
            choices=cls.inputtypes.keys(),
            default='keras'
        )
        group.add_argument(
            '--cluster-dense',
            help='Clusterize only dense layers',
            action='store_true'
        )
        group.add_argument(
            '--clusters-number',
            help='Number of cluster centroids which split each layer.',
            type=int,
            default=10
        )
        group.add_argument(
            '--disable-sparsity-preservation',
            help="Do not preserve sparsity of a given model",
            action='store_false',
        )
        return parser, group

    @classmethod
    def from_argparse(cls, dataset, args):
        return cls(
            dataset,
            args.compiled_model_path,
            args.model_framework,
            args.cluster_dense,
            args.clusters_number,
            args.disable_sparsity_preservation
        )

    def compile(
            self,
            inputmodelpath: Path,
            inputshapes: Dict[str, Tuple[int, ...]],
            dtype: str = 'float32'):
        model = self.inputtypes[self.inputtype](inputmodelpath)
        self.get_inputdtype = dtype

        clustering_params = {
            'number_of_clusters': self.clusters_number,
            'cluster_centroids_init':
                tfmot.clustering.keras.CentroidInitialization.KMEANS_PLUS_PLUS,
            'preserve_sparsity': not self.disable_sparsity_preservation
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

        optimized_model = tfmot.clustering.keras.strip_clustering(
            clustered_model
        )

        optimized_model.save(
            self.compiled_model_path,
            include_optimizer=False,
            save_format='h5'
        )

    def get_framework_and_version(self):
        return ('tensorflow', tf.__version__)

    def get_inputdtype(self):
        return self.inputdtype
