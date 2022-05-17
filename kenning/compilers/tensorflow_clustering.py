"""
Wrapper for TensorFlowClustering optimizer.
"""
import tensorflow as tf
from pathlib import Path
from typing import Dict, Tuple
import tensorflow_model_optimization as tfmot

from kenning.core.optimizer import Optimizer
from kenning.core.dataset import Dataset
from kenning.utils.args_manager import add_parameterschema_argument, add_argparse_argument  # noqa: E501


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

    arguments_structure = {
        'modelframework': {
            'argparse_name': '--model-framework',
            'description': 'The input type of the model, framework-wise',
            'default': 'keras',
            'enum': list(inputtypes.keys()),
        },
        'cluster_dense': {
            'argparse_name': '--cluster-dense',
            'description': 'Clusterize only dense layers',
            'type': bool,
            'default': False,
        },
        'clusters_number': {
            'argparse_name': '--clusters-number',
            'description': 'Number of cluster centroids that split each layer',
            'type': int,
            'default': 10,
        },
        'disable_sparsity_preservation': {
            'argparse_name': '--disable-sparsity-preservation',
            'description': 'Disable sparsity preservation of a given model',
            'type': bool,
            'default': True,
        }
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: Path,
            modelframework: str = 'keras',
            cluster_dense: bool = 'False',
            clusters_number: int = '10',
            disable_sparsity_preservation: bool = 'True'):
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
        parser, group = super().form_argparse(quantizes_model=True)
        add_argparse_argument(
            group,
            TensorFlowClusteringOptimizer.arguments_structure
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

    @classmethod
    def form_parameterschema(cls):
        parameterschema = super().form_parameterschema(quantizes_model=False)
        add_parameterschema_argument(
            parameterschema,
            TensorFlowClusteringOptimizer.arguments_structure
        )
        return parameterschema

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
