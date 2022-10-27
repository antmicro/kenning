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

        super.__init__(
            modelpath,
            dataset,
            from_file,
            (tf.TensorSpec(1, 128, 3, 1, name='input_1'),),
        )

    def get_input_spec(self):
        return {'input_1': (1, 128, 3, 1)}, 'float32'

    def load_model(self, modelpath):
        # https://github.com/tensorflow/tflite-micro/blob/dde75de483faa8d5e42b875cef3aaf26f6c63101/tensorflow/lite/micro/examples/magic_wand/train/train.py#L51
        self.keras_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                8,
                (4, 3),
                padding="same",
                activation="relu",
                input_shape=(self.dataset.window_size, 3, 1)),
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
            self.keras_model.load_weights(self.modelpath)
        else:
            self.train_model()
        converter = tf.lite.TFLiteConverter.from_keras_model(self.keras_model)
        if self.quantize_model:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.model = converter.convert()

    def save_model(self, modelpath):
        open(modelpath, 'wb').write(self.model)

    def train_model(
            self,
            batch_size=64,
            learning_rate=0.99,
            epochs=50,
            logdir=None):
        self.keras_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=['accuracy']
        )
# TODO: Do train / test / validation splits from the dataset
# and check how the training data looks when it is shaped
        train_data = None
        test_data = None
        valid_data = None
        test_labels = np.zeros(len(test_data))
        idx = 0
        for data, label in test_data:
            test_labels[idx] = label.numpy()
            idx += 1
        train_data = train_data.batch(batch_size).repeat()
        valid_data = valid_data.batch(batch_size)
        test_data = test_data.batch(batch_size)
        self.keras_model.fit(
            train_data,
            epochs=epochs,
            validation_data=valid_data,
            steps_per_epoch=1000,
            validation_steps=int((len(valid_data) - 1) / batch_size + 1),
            callbacks=[tf.keras.callbacks.TensorBoard(log_dir=logdir)])
        loss, acc = self.keras_model.evaluate(test_data)
        pred = np.argmax(self.keras_model.predict(test_data), axis=1)
        confusion = tf.math.confusion_matrix(labels=tf.constant(test_labels),
                                             predictions=tf.constant(pred),
                                             num_classes=4)
        print(confusion)
        print("Loss {}, Accuracy {}".format(loss, acc))
