"""
Contains Tensorflow models for the pet classification.

Pretrained on ImageNet dataset, trained on Pet Dataset
"""

from pathlib import Path
import tensorflow as tf

from edge_ai_tester.core.dataset import Dataset
from edge_ai_tester.modelwrappers.frameworks.tensorflow import TensorFlowWrapper  # noqa: E501


class TensorFlowPetDatasetMobileNetV2(TensorFlowWrapper):
    def __init__(self, modelpath: Path, dataset: Dataset, from_file=True):
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        self.numclasses = dataset.numclasses
        self.mean, self.std = dataset.get_input_mean_std()
        super().__init__(
            modelpath,
            dataset,
            from_file,
            (tf.TensorSpec((1, 224, 224, 3), name='input_1'),)
        )

    def get_input_spec(self):
        return {'input_1': (1, 224, 224, 3)}, 'float32'

    def prepare_model(self):
        if self.from_file:
            self.load_model(self.modelpath)
        else:
            self.base = tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
            self.base.trainable = False
            avgpool = tf.keras.layers.GlobalAveragePooling2D()(
                self.base.output
            )
            layer1 = tf.keras.layers.Dense(
                1024,
                activation='relu')(avgpool)
            d1 = tf.keras.layers.Dropout(0.5)(layer1)
            layer2 = tf.keras.layers.Dense(
                512,
                activation='relu')(d1)
            d2 = tf.keras.layers.Dropout(0.5)(layer2)
            layer3 = tf.keras.layers.Dense(
                128,
                activation='relu')(d2)
            d3 = tf.keras.layers.Dropout(0.5)(layer3)
            output = tf.keras.layers.Dense(
                self.numclasses,
                name='out_layer'
            )(d3)
            self.model = tf.keras.models.Model(
                inputs=self.base.input,
                outputs=output
            )
            print(self.model.summary())

    def train_model(
            self,
            batch_size: int,
            learning_rate: int,
            epochs: int,
            logdir: Path):

        def preprocess_input(path, onehot):
            data = tf.io.read_file(path)
            img = tf.io.decode_jpeg(data, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = tf.image.convert_image_dtype(img, dtype=tf.float32)
            img /= 255.0
            img = tf.image.random_brightness(img, 0.1)
            img = tf.image.random_contrast(img, 0.7, 1.0)
            img = tf.image.random_flip_left_right(img)
            img = (img - self.mean) / self.std
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
        ).batch(batch_size)
        validdataset = tf.data.Dataset.from_tensor_slices((Xv, Yv))
        validdataset = validdataset.map(
            preprocess_input,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(batch_size)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            str(logdir),
            histogram_freq=1
        )

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(logdir),
            monitor='val_categorical_accuracy',
            mode='max',
            save_best_only=True
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy()
            ]
        )

        self.model.fit(
            traindataset,
            epochs=epochs,
            callbacks=[
                tensorboard_callback,
                model_checkpoint_callback
            ],
            validation_data=validdataset
        )
