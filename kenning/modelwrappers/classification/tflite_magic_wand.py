import tensorflow as tf
import numpy as np
from pathlib import Path

from kenning.modelwrappers.frameworks.tensorflow import TensorFlowWrapper
from kenning.core.dataset import Dataset


class MagicWandModelWrapper(TensorFlowWrapper):
    def __init__(
            self,
            modelpath: Path,
            dataset: Dataset,
            from_file: bool):
        self.modelpath = modelpath
        self.dataset = dataset
        self.from_file = from_file
        self.numclasses = len(self.dataset.get_class_names())

        super().__init__(
            modelpath,
            dataset,
            from_file,
            (tf.TensorSpec((1, 128, 3, 1), name='input_1'),)
        )

    def get_input_spec(self):
        return {'input_1': (1, 128, 3, 1)}, 'float32'

    def prepare_model(self):
        # https://github.com/tensorflow/tflite-micro/blob/dde75de483faa8d5e42b875cef3aaf26f6c63101/tensorflow/lite/micro/examples/magic_wand/train/train.py#L51
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                8,
                (4, 3),
                padding="same",
                activation="relu",
                input_shape=(128, 3, 1)),
            tf.keras.layers.MaxPool2D((3, 3)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Conv2D(
                16,
                (4, 1),
                padding="same",
                activation="relu"),
            tf.keras.layers.MaxPool2D((3, 1), padding="same"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(4, activation="softmax")
        ])

        if self.from_file:
            self.model.load_weights(self.modelpath)
        else:
            self.train_model()

    def save_model(self, modelpath):
        open(modelpath, 'wb').write(self.model)

    def train_model(
            self,
            batch_size=64,
            learning_rate=0.99,
            epochs=50,
            logdir="/tmp/tflite_magic_wand_logs"):
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy']
        )
        train_data, test_data,\
            train_labels, test_labels,\
            validation_data, validation_labels = \
            self.dataset.train_test_split_representations(validation=True)
        for i, data in enumerate(train_data):
            train_data[i] = self.dataset.split_sample_to_windows(
                self.dataset.generate_padding(data)
            )
        for i, data in enumerate(test_data):
            test_data[i] = self.dataset.split_sample_to_windows(
                self.dataset.generate_padding(data)
            )
        for i, data in enumerate(validation_data):
            validation_data[i] = self.dataset.split_sample_to_windows(
                self.dataset.generate_padding(data)
            )
        train_data = self.dataset.prepare_tf_dataset(
            train_data,
            train_labels
        ).batch(batch_size).repeat()
        valid_data = self.dataset.prepare_tf_dataset(
            validation_data,
            validation_labels
        ).batch(batch_size)
        test_data = self.dataset.prepare_tf_dataset(
            test_data,
            test_labels
        ).batch(batch_size)
        self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=valid_data,
            steps_per_epoch=1000,
            validation_steps=int((len(valid_data) - 1) / batch_size + 1),
            callbacks=[tf.keras.callbacks.TensorBoard(log_dir=logdir)])
        loss, acc = self.model.evaluate(test_data)
        pred = np.argmax(self.model.predict(test_data), axis=1)
        confusion = tf.math.confusion_matrix(
            labels=tf.constant(test_labels),
            predictions=tf.constant(pred),
            num_classes=4
        )
        print(confusion)
        print("Loss {}, Accuracy {}".format(loss, acc))
        self.model.save(self.modelpath)