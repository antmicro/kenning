"""
Wrapper for TensorFlow optimizers
"""

from pathlib import Path
from typing import List, Tuple, Optional
import tensorflow as tf

from kenning.utils.args_manager import add_parameterschema_argument, add_argparse_argument  # noqa: E501
from kenning.core.optimizer import Optimizer
from kenning.core.dataset import Dataset


class TensorFlowOptimizer(Optimizer):

    arguments_structure = {
        'epochs': {
            'description': 'Number of epochs for the training',
            'type': int,
            'default': 3
        },
        'batch_size': {
            'description': 'The size of a batch for the training',
            'type': int,
            'default': 32
        },
        'optimizer': {
            'description': 'Optimizer used during the training',
            'type': str,
            'default': 'adam',
            'enum': ['adam', 'SGD', 'RMSprop']
        },
        'disable_from_logits': {
            'description': 'Determines whether output of the model is normalized',  # noqa: E501
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
            disable_from_logits: bool = False):
        """
        TensorFlowOptimizer framework.

        This class adds a functionality for models fine-tuning.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the model - may be used for quantization
            or fine-tuning
        compiled_model_path : Path
            Path where compiled model will be saved
        epochs : int
            Number of epochs used to fine-tune the model
        batch_size : int
            The size of a batch used for the fine-tuning
        optimizer : str
            Optimizer used during the training
        disable_from_logits
            Determines whether output of the model is normalized
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.disable_from_logits = disable_from_logits
        super().__init__(dataset, compiled_model_path)

    @classmethod
    def form_parameterschema(cls):
        parameterschema = super(
            TensorFlowOptimizer,
            TensorFlowOptimizer
        ).form_parameterschema()

        if cls.arguments_structure != TensorFlowOptimizer.arguments_structure:
            add_parameterschema_argument(
                parameterschema,
                cls.arguments_structure
            )
        return parameterschema

    @classmethod
    def form_argparse(cls):
        parser, group = super(
            TensorFlowOptimizer,
            TensorFlowOptimizer
        ).form_argparse()

        if cls.arguments_structure != TensorFlowOptimizer.arguments_structure:
            add_argparse_argument(
                group,
                cls.arguments_structure
            )
        return parser, group

    def prepare_train_validation(self) -> Tuple:
        """
        Prepares train and validation datasets of the model
        and splits them into batches.

        Returns
        -------
        Tuple : Batched train and validation datasets
        """
        Xt, Xv, Yt, Yv = self.dataset.train_test_split_representations()

        Xt = self.dataset.prepare_input_samples(Xt)
        Yt = self.dataset.prepare_output_samples(Yt)
        traindataset = tf.data.Dataset.from_tensor_slices((Xt, Yt))
        traindataset = traindataset.batch(
            self.batch_size,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        Xv = self.dataset.prepare_input_samples(Xv)
        Yv = self.dataset.prepare_output_samples(Yv)
        validdataset = tf.data.Dataset.from_tensor_slices((Xv, Yv))
        validdataset = validdataset.batch(
            self.batch_size,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return traindataset, validdataset

    def train_model(self, model, callbacks: Optional[List] = None):
        """
        Compiles and trains the given model.

        The function can be used to retrain the model if needed.

        Parameters
        ----------
        model
            The keras model to retrain
        callbacks : Optional[List]
            List of callback function to use during the training.

        Returns
        -------
        Trained keras model
        """
        traindataset, validdataset = self.prepare_train_validation()

        model.compile(
            optimizer=self.optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(
                from_logits=not self.disable_from_logits
            ),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy()
            ]
        )

        model.fit(
            traindataset,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=1,
            validation_data=validdataset
        )

        return model

    def get_framework_and_version(self):
        return ('tensorflow', tf.__version__)
