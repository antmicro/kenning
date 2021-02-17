"""
Contains Tensorflow models for the pet classification.

Pretrained on ImageNet dataset, trained on Pet Dataset
"""

from dl_framework_analyzer.core.model import ModelWrapper
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from dl_framework_analyzer.core.dataset import Dataset


class TensorflowPetDatasetMobileNetV2(ModelWrapper):
    def __init__(self, dataset: Dataset):
        self.numclasses = dataset.numclasses
        super().__init__(dataset)

    def prepare_model(self):
        self.base = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        avgpool = tf.keras.layers.GlobalAveragePooling2D()(self.base.output)
        output = tf.keras.layers.Dense(
            self.numclasses,
            name='out_layer'
        )(avgpool)
        self.model = tf.keras.models.Model(
            inputs=self.base.input,
            outputs=output
        )
        print(self.model.summary())

    def run_inference(self, X):
        return self.model.predict(X)

    def get_framework_and_version(self):
        return ('tensorflow', tf.__version__)

    def train_model(self, outputmodel, logdir=None):
        BATCH_SIZE = 64
        LEARNING_RATE = 0.0001
        EPOCHS = 20

        def preprocess_input(path, onehot):
            data = tf.io.read_file(path)
            img = tf.io.decode_jpeg(data, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = tf.image.convert_image_dtype(img, dtype=tf.float32)
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.7, 1.0)
            img = tf.image.random_flip_left_right(img)
            img = tfa.image.rotate(
                img,
                tf.random.uniform([], minval=-0.3, maxval=0.3), 'BILINEAR'
            )
            return img, tf.convert_to_tensor(onehot)
        
        Xt, Xv, Yt, Yv = self.dataset.train_test_split_representations(
            0.25
        )
        Yt = self.dataset.prepare_output_samples(Yt)
        Yv = self.dataset.prepare_output_samples(Yv)
        traindataset = tf.data.Dataset.from_tensor_slices((Xt, Yt))
        traindataset = traindataset.map(
            preprocess_input,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(BATCH_SIZE)
        validdataset = tf.data.Dataset.from_tensor_slices((Xv, Yv))
        validdataset = validdataset.map(
            preprocess_input,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(BATCH_SIZE)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            logdir,
            histogram_freq=1
        )

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=logdir,
            monitor='val_categorical_accuracy',
            mode='max',
            save_best_only=True
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(),
                tfa.metrics.MultiLabelConfusionMatrix(
                    num_classes=self.numclasses
                )
            ]
        )

        self.model.fit(
            traindataset,
            epochs = EPOCHS,
            callbacks = [
                tensorboard_callback,
                model_checkpoint_callback
            ],
            validation_data = validdataset
        )
