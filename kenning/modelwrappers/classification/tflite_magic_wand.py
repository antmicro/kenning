from typing import Dict, List
import tensorflow as tf
import numpy as np
from pathlib import Path
import sys
if sys.version_info.minor < 9:
    from importlib_resources import files
else:
    from importlib.resources import files

from kenning.modelwrappers.frameworks.tensorflow import TensorFlowWrapper
from kenning.core.dataset import Dataset
from kenning.utils.logger import get_logger
from kenning.datasets.magic_wand_dataset import MagicWandDataset
from kenning.resources.models import classification


class MagicWandModelWrapper(TensorFlowWrapper):

    default_dataset = MagicWandDataset
    pretrained_modelpath = files(classification) / 'magic_wand.h5'
    arguments_structure = {
        'window_size': {
            'argparse_name': '--window-size',
            'description': 'Determines the size of single sample window',
            'default': 128,
        }
    }

    def __init__(
            self,
            modelpath: Path,
            dataset: Dataset,
            from_file: bool,
            window_size: int = 128):
        """
        Creates the Magic Wand model wrapper.

        Parameters
        ----------
        modelpath : Path
            The path to the model
        dataset : Dataset
            The dataset to verify the inference
        from_file : bool
            True if the model should be loaded from file
        windows_size : int
            Size of single sample window
        """
        super().__init__(
            modelpath,
            dataset,
            from_file
        )
        self.window_size = window_size
        if dataset is not None:
            self.class_names = self.dataset.get_class_names()
            self.numclasses = len(self.class_names)
            self.save_io_specification(self.modelpath)

    @classmethod
    def from_argparse(cls, dataset, args, from_file=False):
        return cls(
            args.modelpath,
            dataset,
            from_file,
            args.window_size
        )

    def get_io_specification_from_model(self) -> Dict[str, List[Dict]]:
        return {
            'input': [{
                'name': 'input_1',
                'shape': (1, self.window_size, 3, 1),
                'dtype': 'float32'
            }],
            'output': [{
                'name': 'out_layer',
                'shape': (1, self.numclasses),
                'dtype': 'float32',
                'class_names': self.class_names
            }]
        }

    def prepare_model(self):
        if self.model_prepared:
            return None
        # https://github.com/tensorflow/tflite-micro/blob/dde75de483faa8d5e42b875cef3aaf26f6c63101/tensorflow/lite/micro/examples/magic_wand/train/train.py#L51
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(
                input_shape=(self.window_size, 3, 1),
                name='input_1'
            ),
            tf.keras.layers.Conv2D(
                8,
                (4, 3),
                padding='same',
                activation='relu'),
            tf.keras.layers.MaxPool2D((3, 3)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv2D(
                16,
                (4, 1),
                padding='same',
                activation='relu'),
            tf.keras.layers.MaxPool2D((3, 1), padding='same'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(4, activation='softmax', name='out_layer')
        ])
        self.model.summary()

        if self.from_file:
            self.model.load_weights(self.modelpath)
            self.model_prepared = True
        else:
            self.train_model()
            self.model_prepared = True
            self.save_model(self.modelpath)

    def train_model(
            self,
            batch_size=64,
            learning_rate=0.001,
            epochs=50,
            logdir='/tmp/tflite_magic_wand_logs'):
        log = get_logger()
        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        train_data, test_data,\
            train_labels, test_labels,\
            val_data, val_labels = \
            self.dataset.train_test_split_representations(validation=True)

        train_dataset = self.dataset.prepare_tf_dataset(
            train_data,
            train_labels
        ).batch(batch_size).repeat()
        val_dataset = self.dataset.prepare_tf_dataset(
            val_data,
            val_labels
        ).batch(batch_size)
        test_dataset = self.dataset.prepare_tf_dataset(
            test_data,
            test_labels
        ).batch(batch_size)
        test_labels = np.concatenate([y for _, y in test_dataset], axis=0)

        self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            steps_per_epoch=1000,
            validation_steps=int((len(val_data) - 1) / batch_size + 1),
            callbacks=[tf.keras.callbacks.TensorBoard(log_dir=logdir)]
        )
        loss, acc = self.model.evaluate(test_dataset)
        pred = np.argmax(self.model.predict(test_dataset), axis=1)
        confusion = tf.math.confusion_matrix(
            labels=tf.constant(test_labels),
            predictions=tf.constant(pred),
            num_classes=4
        )
        log.info(confusion)
        log.info(f'loss: {loss}, accuracy: {acc}')
        self.model.save(self.modelpath)
