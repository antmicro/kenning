"""
Wrapper for TensorFlowPruning optimizer.
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


class TensorFlowPruningOptimizer(Optimizer):
    """
    The TensorFlowPruning optimizer.
    """
    outputtypes = [
        'keras'
    ]

    inputtypes = {
        'keras': kerasconversion,
    }

    arguments_structure = {
        'modelframework': {
            'argparse_name': '--model-framework',
            'description': 'The input type of the model, framework-wise',
            'default': 'keras',
            'enum': list(inputtypes.keys())
        },
        'epochs': {
            'description': 'Number of epochs for the fine-tuning',
            'type': int,
            'default': 10
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
        'batch_size': {
            'description': 'The size of a batch for the fine-tuning',
            'type': int,
            'default': 32
        }
    }

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: Path,
            modelframework: str = 'keras',
            epochs: int = 10,
            prune_dense: bool = False,
            target_sparsity: float = 0.1,
            batch_size: int = 32):
        """
        The TensorFlowPruning optimizer.

        This compiler applies pruning optimization to the model.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the model - may be used for quantization
            during compilation stage
        compiled_model_path : Path
            Path where compiled model will be saved
        modelframework : str
            Framework of the input model, used to select a proper backend
        epochs : int
            Number of epochs used to fine-tune the model
        prune_dense : bool
            Determines if only dense layers should be pruned
        target_sparsity : float
            Target weights sparsity of the model after pruning
        batch_size : int
            The size of a batch used for the fine-tuning
        """
        self.modelframework = modelframework
        self.epochs = epochs
        self.prune_dense = prune_dense
        self.target_sparsity = target_sparsity
        self.batch_size = batch_size
        self.set_input_type(modelframework)
        super().__init__(dataset, compiled_model_path)

    @classmethod
    def form_argparse(cls):
        parser, group = super().form_argparse(quantizes_model=True)
        add_argparse_argument(
            group,
            TensorFlowPruningOptimizer.arguments_structure
        )
        return parser, group

    @classmethod
    def from_argparse(cls, dataset, args):
        return cls(
            dataset,
            args.compiled_model_path,
            args.model_framework,
            args.epochs,
            args.prune_dense,
            args.target_sparsity,
            args.batch_size
        )

    @classmethod
    def form_parameterschema(cls):
        parameterschema = super().form_parameterschema(quantizes_model=False)
        add_parameterschema_argument(
            parameterschema,
            TensorFlowPruningOptimizer.arguments_structure
        )
        return parameterschema

    def compile(
            self,
            inputmodelpath: Path,
            inputshapes: Dict[str, Tuple[int, ...]],
            dtype: str = 'float32'):

        model = self.inputtypes[self.inputtype](inputmodelpath)
        self.inputdtype = dtype

        def preprocess_output(input, output):
            return input, tf.convert_to_tensor(output)

        Xt, Xv, Yt, Yv = self.dataset.train_test_split_representations()

        Xt = self.dataset.prepare_input_samples(Xt)
        Yt = self.dataset.prepare_output_samples(Yt)
        traindataset = tf.data.Dataset.from_tensor_slices((Xt, Yt))
        traindataset = traindataset.map(
            preprocess_output,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(self.batch_size)

        Xv = self.dataset.prepare_input_samples(Xv)
        Yv = self.dataset.prepare_output_samples(Yv)
        validdataset = tf.data.Dataset.from_tensor_slices((Xv, Yv))
        validdataset = validdataset.map(
            preprocess_output,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(self.batch_size)

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

            model_for_pruning = tf.keras.models.clone_model(
                model,
                clone_function=apply_pruning_to_dense,
            )
        else:
            model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
                model,
                **pruning_params
            )

        model_for_pruning.compile(
            optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy()
            ]
        )

        model_for_pruning.fit(
            traindataset,
            epochs=self.epochs,
            callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
            verbose=1,
            validation_data=validdataset
        )

        optimized_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
        optimized_model.save(
            self.compiled_model_path,
            include_optimizer=False,
            save_format='h5'
        )

    def get_framework_and_version(self):
        return ('tensorflow', tf.__version__)
